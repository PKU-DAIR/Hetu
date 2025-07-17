from concurrent import futures
import logging
import multiprocessing.spawn

import grpc
import heturpc_pb2
import heturpc_pb2_grpc

import threading
import time
import multiprocessing
import argparse
from pssh_workers import *
import paramiko
logging.getLogger("paramiko").setLevel(logging.WARNING)
import csv
import shlex
import json
import enum
from elastic_arg_parser import ElasticStrategy, \
                               replace_argument, get_argument, add_argument, delete_argument
from utils import read_yaml

MAX_UNFOUND_TIMES = 10000
MAX_WORKERS = 64
MAX_GPUS = 1024
DEVICES_PER_NODE = 8
TIMEOUT=5.0

BARRIER_DICT = {79:"SAVING_BARRIER_VALUE"}



class SERVER_STATUS(Enum):
    EXIT = 0       # all rank exit 
    ERROR = 1      # at least one rank error
    RESTART = 2    # num gpus changed

class DeviceController(heturpc_pb2_grpc.DeviceControllerServicer):
    # def __init__(self, arr, exit_arr, last_heartbeat, pssh_pool:PSSHHandlerPool) -> None:
    def __init__(self, arrs, pssh_pool:PSSHHandlerPool) -> None:
        super().__init__()
        self.arr = arrs['arr']
        self.exit_arr = arrs['exit_arr']
        self.error_arr = arrs['error_arr']
        self.last_heartbeat = arrs['last_heartbeat']
        self.stop = arrs['stop']
        self.worker_terminal = arrs['worker_terminal']
        self.host_name_to_idx = arrs['host_name_to_idx']
        self.host_name_to_local_idx = arrs['host_name_to_local_idx']
        self.disabled_num_ranks = arrs['disabled_num_ranks']
        # self.device_to_rank = arrs['']
        self.pssh_pool = pssh_pool
        #locks
        self.lock = threading.Lock()
        self.saving_lock = threading.Lock()
        self.double_lock = threading.Lock()
        self.int_lock = threading.Lock()
        self.string_lock = threading.Lock()
        self.bytes_lock = threading.Lock()
        self.json_lock = threading.Lock()
        # for start
        self.worldsize = 0
        self.local_worldsizes = {}
        self.local_ranks = {}
        self.nodenames = []
        self.hostnames = {}
        self.num_hostnames = 0
        self.deviceinfos = {}
        self.num_deviceinfos = 0
        self.nccl_ids = {}
        self.exit_nums = 0
        self.barrier_nums = {}
        self.barrier_end_nums = {}
        self.consistent_value = {}
        #dicts
        self.double_dict = {}
        self.int_dict = {}
        self.string_dict = {}
        self.bytes_dict = {}
        self.json_dict = {}
        #default value
        self.int_dict["STOP"] = 0
        self.max_load_t = 0
        self.max_load_m = 0

    def Connect(self, request, context):
        self.lock.acquire()
        nodename = "node-" + request.hostname
        if nodename not in self.local_worldsizes:
            self.local_worldsizes[nodename] = 0
            self.nodenames.append(nodename)
        self.local_worldsizes[nodename] = self.local_worldsizes[nodename] + 1
        self.worldsize += 1
        print("connect num:", self.worldsize, self.local_worldsizes)
        self.lock.release()
        return heturpc_pb2.ConnectReply(status=1)
        
    def GetRank(self, request, context):
        local_rank = 0
        self.lock.acquire()
        if self.host_name_to_idx == {}:
            nodename = "node-" + request.name
            if nodename not in self.local_ranks:
                self.local_ranks[nodename] = 0
            for nodename_ in self.nodenames:
                if nodename_ == nodename:
                    local_rank += self.local_ranks[nodename]
                    self.local_ranks[nodename] = self.local_ranks[nodename] + 1
                    break
                else:
                    local_rank += self.local_worldsizes[nodename_]
        else:
            nodename = request.name
            assert(request.name in self.host_name_to_idx)
            if nodename not in self.local_ranks:
                self.local_ranks[nodename] = 0
            local_rank = self.host_name_to_idx[nodename][self.local_ranks[nodename]]
            local_device = self.host_name_to_local_idx[nodename][self.local_ranks[nodename]]
            self.local_ranks[nodename] = self.local_ranks[nodename] + 1

        # print(request.name, " ", local_rank)
        print(request.name, "rank", local_rank, "confirm")
        self.last_heartbeat[int(local_rank)] = time.time()
        self.arr[0] += 1
        self.lock.release()
        return heturpc_pb2.RankReply(rank=local_rank, local_device=local_device)
    
    def CommitHostName(self, request, context):
        self.lock.acquire()
        self.num_hostnames += 1
        self.hostnames[request.rank] = request.hostname
        # print(f"RANK:{request.rank} COMMIT HOSTNAME:{request.hostname} "
        #       f"HOSTNAMES:{self.hostnames}")
        self.lock.release()
        return heturpc_pb2.CommitHostNameReply(status=1)

    def GetHostName(self, request, context):
        # print(f"GETHOST RANK:{request.rank} DISABLE:{self.disabled_num_ranks}")
        if request.rank in self.disabled_num_ranks:
            return heturpc_pb2.GetHostNameReply(hostname="none")
        while(request.rank not in self.hostnames):
            time.sleep(0.0001)
        self.lock.acquire()
        hostname = self.hostnames[request.rank]
        self.lock.release()
        return heturpc_pb2.GetHostNameReply(hostname=hostname)

    def CommitDeviceInfo(self, request, context):
        self.lock.acquire()
        self.num_deviceinfos += 1
        self.deviceinfos[request.rank] = (request.type, request.index, request.multiplex)
        self.lock.release()
        return heturpc_pb2.CommitDeviceInfoReply(status=1)

    def GetDeviceInfo(self, request, context):
        if request.rank in self.disabled_num_ranks:
            return heturpc_pb2.GetDeviceInfoReply(type=0, index=0, multiplex=0)
        while(request.rank not in self.deviceinfos):
            time.sleep(0.0001)
        self.lock.acquire()
        mtype, mindex, mmultiplex = self.deviceinfos[request.rank]
        self.lock.release()
        return heturpc_pb2.GetDeviceInfoReply(type=mtype, index=mindex, multiplex=mmultiplex)

    def CommitNcclId(self, request, context):
        self.lock.acquire()
        world_rank = tuple(request.world_rank) + (int(request.stream_id),)
        print("GNCCL COMMIT:", world_rank, type(world_rank))
        self.nccl_ids[world_rank] = request.nccl_id
        self.lock.release()
        return heturpc_pb2.CommitNcclIdReply(status=1)

    def GetNcclId(self, request, context):
        world_rank = tuple(request.world_rank) + (int(request.stream_id),)
        while(world_rank not in self.nccl_ids):
            time.sleep(0.0001)
        self.lock.acquire()
        nccl_id = self.nccl_ids[world_rank]
        print("GNCCL RECEIVE:", world_rank, type(world_rank))
        self.lock.release()
        return heturpc_pb2.GetNcclIdReply(nccl_id=nccl_id)

    def Exit(self, request, context):
        self.lock.acquire()
        self.exit_arr[request.rank] = 1
        self.last_heartbeat[request.rank] = -1
        self.exit_nums += 1
        self.lock.release()
        print(self.exit_nums, " ", self.worldsize)
        # while(self.exit_nums != self.worldsize):
        #     time.sleep(0.0001)
        # self.exit_arr[request.rank] = 1
        return heturpc_pb2.ExitReply(status=1)

    def PutDouble(self, request, context):
        self.double_lock.acquire()
        self.double_dict[request.key] = request.value
        print("DOUBLE key:", request.key, ",value:", request.value)
        if request.key == "LOAD_T":
            self.max_load_t = max(self.max_load_t, float(request.value))
            print(f"RECEIVE_T:{float(request.value)}, MAX_T:{self.max_load_t}")
        if request.key == "LOAD_M":
            self.max_load_m = max(self.max_load_m, float(request.value))
            print(f"RECEIVE_M:{float(request.value)}, MAX_M:{self.max_load_m}")
        self.double_lock.release()
        return heturpc_pb2.PutDoubleReply(status=1)

    def GetDouble(self, request, context):
        while(request.key not in self.int_dict):
            time.sleep(0.0001)
        self.double_lock.acquire()
        value = self.double_dict[request.key]
        self.int_lock.release()
        return heturpc_pb2.GetDoubleReply(value=value)

    def RemoveDouble(self, request, context):
        unfound_times = 0
        while(request.key not in self.double_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveDoubleReply(message="not found:" + request.key)
        else:
            self.double_dict.pop(request.key)
        return heturpc_pb2.RemoveDoubleReply(message="already remove:" + request.key)
    
    def PutInt(self, request, context):
        self.int_lock.acquire()
        self.int_dict[request.key] = request.value
        self.int_lock.release()
        return heturpc_pb2.PutIntReply(status=1)

    def GetInt(self, request, context):
        while(request.key not in self.double_dict):
            time.sleep(0.0001)
        self.double_lock.acquire()
        value = self.double_dict[request.key]
        self.double_lock.release()
        return heturpc_pb2.GetIntReply(value=value)

    def RemoveInt(self, request, context):
        unfound_times = 0
        while(request.key not in self.int_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveIntReply(message="not found:" + request.key)
        else:
            self.int_dict.pop(request.key)
        return heturpc_pb2.RemoveIntReply(message="already remove:" + request.key)
    
    def PutString(self, request, context):
        self.string_lock.acquire()
        self.string_dict[request.key] = request.value
        print("PUT key:", request.key, ",value:", request.value)
        self.string_lock.release()
        return heturpc_pb2.PutStringReply(status=1)

    def GetString(self, request, context):
        while(request.key not in self.string_dict):
            time.sleep(0.0001)
        self.string_lock.acquire()
        value = self.string_dict[request.key]
        self.string_lock.release()
        return heturpc_pb2.GetStringReply(value=value)

    def RemoveString(self, request, context):
        unfound_times = 0
        while(request.key not in self.string_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveStringReply(message="not found:" + request.key)
        else:
            self.string_dict.pop(request.key)
        return heturpc_pb2.RemoveStringReply(message="already remove:" + request.key)
    
    def PutBytes(self, request, context):
        self.bytes_lock.acquire()
        self.bytes_dict[request.key] = request.value
        self.bytes_lock.release()
        return heturpc_pb2.PutBytesReply(status=1)

    def GetBytes(self, request, context):
        while(request.key not in self.bytes_dict):
            time.sleep(0.0001)
        self.bytes_lock.acquire()
        value = self.bytes_dict[request.key]
        self.bytes_lock.release()
        return heturpc_pb2.GetBytesReply(value=value)
    
    def RemoveBytes(self, request, context):
        unfound_times = 0
        while(request.key not in self.bytes_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveBytesReply(message="not found:" + request.key)
        else:
            self.bytes_dict.pop(request.key)
        return heturpc_pb2.RemoveBytesReply(message="already remove:" + request.key)

    def PutJson(self, request, context):
        self.json_lock.acquire()
        self.json_dict[request.key] = request.value
        import json
        my_json = json.loads(request.value)
        # print("ReCEIVEJSON:", my_json)
        self.json_lock.release()
        return heturpc_pb2.PutJsonReply(status=1)

    def GetJson(self, request, context):
        while(request.key not in self.json_dict):
            time.sleep(0.0001)
        self.json_lock.acquire()
        value = self.json_dict[request.key]
        self.json_lock.release()
        return heturpc_pb2.GetJsonReply(value=value)
    
    def RemoveJson(self, request, context):
        unfound_times = 0
        while(request.key not in self.json_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveJsonReply(message="not found:" + request.key)
        else:
            self.json_dict.pop(request.key)
        return heturpc_pb2.RemoveJsonReply(message="already remove:" + request.key)
    
    # def Barrier(self, request, context):
    #     world_rank = tuple(request.world_rank)
    #     self.lock.acquire()
    #     if (world_rank not in self.barrier_nums):
    #         self.barrier_nums[world_rank] = 0
    #     self.lock.release()
    #     self.lock.acquire()
    #     if (request.rank == world_rank[0]):
    #         self.barrier_end_nums[world_rank] = 0
    #     self.barrier_nums[world_rank] += 1
    #     self.lock.release()
    #     # print("S1 Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
    #     while(self.barrier_nums[world_rank] < len(world_rank)):
    #         time.sleep(0.0001)
    #     # print("S2 Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
    #     self.lock.acquire()
    #     self.barrier_end_nums[world_rank] += 1
    #     self.lock.release()
    #     while(self.barrier_end_nums[world_rank] < len(world_rank)):
    #         time.sleep(0.0001)
    #     if (request.rank == world_rank[0]):
    #         self.barrier_nums[world_rank] = 0
    #     # print("End Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
    #     return heturpc_pb2.BarrierReply(status=1)

    def Barrier(self, request, context):
        world_rank = tuple(request.world_rank)
        neglect_ranks = -1
        self.lock.acquire()
        if (world_rank not in self.barrier_nums or self.barrier_nums[world_rank] == -1):
            neglect_ranks = 0
            for rank in world_rank:
                if (rank in self.disabled_num_ranks):
                    neglect_ranks += 1
            self.barrier_nums[world_rank] = neglect_ranks
            print(f"num neglect ranks:{neglect_ranks}, disabled_ranks:{self.disabled_num_ranks}")
        self.lock.release()
        self.lock.acquire()
        if (neglect_ranks >= 0):
            self.barrier_end_nums[world_rank] = neglect_ranks
        self.barrier_nums[world_rank] += 1
        # print(f"BARRIER, LOCAL_RANK:{self.barrier_nums[world_rank]} WORLD_RANK:{world_rank}")
        self.lock.release()
        # print("S1 Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
        while(self.barrier_nums[world_rank] < len(world_rank)):
            time.sleep(0.0001)
        # print("S2 Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
        self.lock.acquire()
        self.barrier_end_nums[world_rank] += 1
        self.lock.release()
        while(self.barrier_end_nums[world_rank] < len(world_rank)):
            time.sleep(0.0001)
        if (neglect_ranks >= 0):
            self.barrier_nums[world_rank] = -1
        # print("End Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
        return heturpc_pb2.BarrierReply(status=1)

    def Consistent(self, request, context):
        value = int(request.value)
        world_rank = tuple(request.world_rank)
        neglect_ranks = -1
        if (value in BARRIER_DICT):
            dst = BARRIER_DICT[value]
            self.saving_lock.acquire()
            if (dst not in self.barrier_nums or self.barrier_nums[dst] == -1):
                neglect_ranks = 0
                self.barrier_nums[dst] = neglect_ranks
                self.consistent_value[dst] = neglect_ranks
                print(f"num neglect ranks:{neglect_ranks}, disabled_ranks:{self.disabled_num_ranks}")
            self.saving_lock.release()
            self.saving_lock.acquire()
            if (neglect_ranks >= 0):
                self.barrier_end_nums[dst] = neglect_ranks
            self.barrier_nums[dst] += 1
            self.consistent_value[dst] += value
            print(f"BARRIER, LOCAL_RANK:{self.barrier_nums[dst]} WORLD_RANK:{dst}")
            self.saving_lock.release()
            print("S1 Barrier:", self.barrier_nums[dst], " ", self.worldsize)
            while(self.barrier_nums[dst] < self.worldsize):
                time.sleep(0.0001)
            print("S2 Barrier:", self.barrier_nums[dst], " ", self.worldsize)
            self.saving_lock.acquire()
            self.barrier_end_nums[dst] += 1
            tmp_consist = self.consistent_value[dst]
            self.saving_lock.release()
            while(self.barrier_end_nums[dst] < self.worldsize):
                time.sleep(0.0001)
            out_value = 0
            if (neglect_ranks >= 0):
                self.barrier_nums[dst] = -1
                self.consistent_value[dst] = -1
            print("End Barrier:", self.barrier_nums[dst], " ", self.worldsize)
            return heturpc_pb2.BarrierReply(status=out_value)
        else:
            self.lock.acquire()
            if (world_rank not in self.barrier_nums or self.barrier_nums[world_rank] == -1):
                neglect_ranks = 0
                for rank in world_rank:
                    if (rank in self.disabled_num_ranks):
                        neglect_ranks += 1
                self.barrier_nums[world_rank] = neglect_ranks
                self.consistent_value[world_rank] = neglect_ranks
                print(f"num neglect ranks:{neglect_ranks}, disabled_ranks:{self.disabled_num_ranks}")
            self.lock.release()
            self.lock.acquire()
            if (neglect_ranks >= 0):
                self.barrier_end_nums[world_rank] = neglect_ranks
            self.barrier_nums[world_rank] += 1
            self.consistent_value[world_rank] += value
            # print(f"BARRIER, LOCAL_RANK:{self.barrier_nums[world_rank]} WORLD_RANK:{world_rank}")
            self.lock.release()
            # print("S1 Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
            while(self.barrier_nums[world_rank] < len(world_rank)):
                time.sleep(0.0001)
            # print("S2 Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
            self.lock.acquire()
            self.barrier_end_nums[world_rank] += 1
            tmp_consist = self.consistent_value[world_rank]
            self.lock.release()
            while(self.barrier_end_nums[world_rank] < len(world_rank)):
                time.sleep(0.0001)
            out_value = 0
            if tmp_consist == len(world_rank):
                out_value = 1
            if (neglect_ranks >= 0):
                self.barrier_nums[world_rank] = -1
                self.consistent_value[world_rank] = -1
            # print("End Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
            # print(f"CONSISTTT, RANK:{request.rank}, VALUE:{value}, WORLD_RANK:{world_rank}")
            return heturpc_pb2.BarrierReply(status=out_value)

    def HeartBeat(self, request, context):
        self.lock.acquire()
        if (request.rank >= 0):
            self.last_heartbeat[request.rank] = time.time()
        self.lock.release()
        return heturpc_pb2.HeartBeatReply(status=1)
    
    def AlreadyStop(self, request, context):
        already_stop = 0
        self.lock.acquire()
        if self.stop[0] > 0:
            already_stop = 1
        self.lock.release()
        return heturpc_pb2.HeartBeatReply(status=already_stop)
    
    def WorkerStop(self, request, context):
        self.lock.acquire()
        if request.rank >= 0:
            self.worker_terminal[request.rank] = 1
            self.stop[0] += 1
            print("rank:", request.rank, "Terminal.")
        self.lock.release()
        return heturpc_pb2.HeartBeatReply(status=0)

# def serve(arr, exit_arr, last_heartbeat, pssh_pool, port):
def serve(arrs, pssh_pool, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    # heturpc_pb2_grpc.add_DeviceControllerServicer_to_server(DeviceController(arr, exit_arr, last_heartbeat, pssh_pool), server)
    heturpc_pb2_grpc.add_DeviceControllerServicer_to_server(DeviceController(arrs, pssh_pool), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()

def detect_node_info(nodes, args=None, shared_dict=None):
    node_info = {}
    for idx, node in enumerate(nodes):
        node_info[idx] = {} 
        node_info[idx]["addr"] = node
        gpu_info = []
        sys_pssh_client = paramiko.SSHClient()
        sys_pssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
        # print(f"NODE:{node}")
        try:
            sys_pssh_client.connect(node, port=args.ssh_port, timeout=TIMEOUT)
            print(f"NODE:{node} connected.")
        except:
            print(f"NODE:{node} disconnected")
            node_info[idx]['gpu_info'] = gpu_info
            node_info[idx]['hostname'] = node_info[idx]['addr']
            sys_pssh_client.close()
            continue

        #Get GPU information
        nvidia_cmd = "nvidia-smi --query-gpu=index,name," + \
                     "memory.total,memory.free,memory.used,utilization.gpu " + \
                     "--format=csv,noheader,nounits"
        _, run_log, error_log = sys_pssh_client.exec_command(nvidia_cmd)
        reader = csv.reader(run_log.read().decode('utf-8').strip().split('\n'))
        hostname_cmd = "hostname"
        _, run_log, error_log = sys_pssh_client.exec_command(hostname_cmd)
        node_info[idx]['hostname'] = run_log.read().decode('utf-8').strip()
        sys_pssh_client.close()
        for row in reader:
            # print("ROW:", row)
            if len(row) == 6:
                device_idx, name, total_memory, free_memory, used_memory, utilization = row
                gpu_info.append({
                    'device_idx': int(device_idx),
                    'name': name,
                    'total_memory': int(total_memory),
                    'free_memory': int(free_memory),
                    'used_memory': int(used_memory),
                    'remain_percent': (int(free_memory) / int(total_memory)),
                    'utilization': float(utilization),
                })
        node_info[idx]['gpu_info'] = gpu_info
    if shared_dict is not None:
        shared_dict['node_info'] = node_info
    return node_info

def detect_gpu_nums(nodes, args=None, shared_dict=None):
    node_info = detect_node_info(nodes, args, shared_dict)
    tmp_es = ElasticStrategy(cmd="", node_info=node_info)
    available_gpus = tmp_es.available_gpu_info()
    available_device_ids = []
    for k, v in available_gpus.items():
        for i in range(v['agpu']):
            available_device_ids.append(v['gpus'][i]['idx'])
    ngpu = tmp_es.get_gpu_num()

    if shared_dict is not None:
      shared_dict['ngpu'] = ngpu
      shared_dict['available_device_ids'] = available_gpus
    return ngpu
    
def server_launch(port, pssh_pool:PSSHHandlerPool, args):
    # recover times
    recover = args.recover
    # recover = 2
    ori_recover = recover

    # get node elastic information
    nodes = []
    if args.hosts:
        host_info = read_yaml(args.hosts)
        max_restart_times = host_info['max_restart_times']
        heartbeat_interval = float(host_info['heartbeat_interval'])
        for host in host_info['hosts']:
            addr = str(host['addr'])
            initial_workers = int(host['initial_workers'])
            min_workers = int(host['min_workers'])
            max_workers = int(host['max_workers'])
            nodes.append(addr)
    else:
        nodes.append("localhost")
        # nodes.append("127.0.0.1")
        # nodes.append("162.105.146.12")

    # alloc gpu process hostnames
    hostnames = []
    if args.hosts is None:
        hostnames = ['localhost'] * args.ngpus
    else:
        host_info = read_yaml(args.hosts)
        max_restart_times = host_info['max_restart_times']
        heartbeat_interval = float(host_info['heartbeat_interval'])
        for host in host_info['hosts']:
            print(host)
            addr = str(host['addr'])
            initial_workers = int(host['initial_workers'])
            min_workers = int(host['min_workers'])
            max_workers = int(host['max_workers'])
            for i in range(initial_workers):
                hostnames.append(addr)
    print(f"HostNames:{hostnames}")
    train_command = args.command
    worker_path = os.environ['HETU_HOME'] + "/python/hetu/rpc/pssh_workers.py"
    cwd = os.getcwd()
    cmd = "cd " + cwd 
    ini_cmd = cmd + f" && source {args.envs}"
    cmd += f" && source {args.envs} && " + "python3 " + worker_path + \
        " --server_addr " + args.server_addr + \
        " --server_port " + args.server_port + \
        " --command \"" + train_command

    # copy environment
    nums = 0
    environ_command = ""
    for k, v in os.environ.items():
        environ_command += f"export {k}=\'{v}\' && "
        nums += 1

    cmd_list = []
    for i in range(len(hostnames)):
        # 请注意log编号目前并不等于rank编号
        # log编号是进程编号
        # 但不能保证分配到同样编号的rank
        # cmd_list.append(cmd + "\"")
        cmd_list.append(cmd + f" 2>&1 | tee {args.log_path}" + "/log_" + f"{i}" + ".txt" + "\"")

    ptr = 0
    pssh_pool.alloc_workers(len(hostnames))
    if args.mpi:
        pssh_pool.register_client(ptr, ['localhost'], 22)
        print(f"MPI CMD:{cmd_list[0]}")
        # pssh_pool.run_command(ptr, cmd_list[0])
        pssh_pool.set_command(ptr, cmd_list[0])
    else:     
        for hostname, cmd in zip(hostnames, cmd_list):
            pssh_pool.set_command(ptr, cmd, ptr)
            ptr += 1

    # the cmd user provided.
    ori_cmd = pssh_pool.get_command(0)

    while recover > 0:
        # recover, try the new strategy
        strategy = None
        node_info = {}
        base_gpu_num = 0
        base_available_device_ids = set()

        if recover <= ori_recover:
            node_info = detect_node_info(nodes, args)
            print(f"NODE_INFO:{node_info}")
            # ori_cmd = pssh_pool.get_command(0)
            # print(f"ORI_CMD:{ori_cmd}")
            num_dead_devices = (ori_recover - recover) % 3
            if (ori_recover - recover == 1):
                num_dead_devices = 1
            elif (ori_recover - recover == 2):
                num_dead_devices = 8
            else:
                num_dead_devices = 0

            num_dead_devices = 0

            for i in range(num_dead_devices):
                node_info[0]['gpu_info'][i]['remain_percent'] = 0
            es = ElasticStrategy(cmd=ori_cmd, node_info=node_info)

            es.parse_node_info()
            parallel_config, strategy = es.search_best_stategy()
            base_gpu_num = es.get_gpu_num()
            print(f"BASE_GPU_NUM:{base_gpu_num}")
            pssh_pool.alloc_workers(base_gpu_num)
            available_gpus = es.available_gpu_info()
            es.generate_parallel_config(args.log_path, ini_cmd)
            ptr = 0
            addresses = []
            for k, v in available_gpus.items():
                addresses.append(v['addr'])
            for k, v in available_gpus.items():
                for i in range(v['agpu']):
                    pssh_pool.register_client(ptr, [v['addr']], int(args.ssh_port))
                    create_new_log = False
                    if recover == ori_recover:
                        create_new_log = True
                    new_command = es.replace_cmd(ori_cmd, create_new_log)
                    new_command = es.renew_step(new_command, 10)
                    node_idx = addresses.index(v['addr'])
                    new_command = es.set_distributed_info(new_command, node_idx, nodes)
                    # if recover == 1:
                    #     new_command = es.set_validation(new_command)
                    # new_command = es.renew_step(new_command, 30)
                    
                    #add environ command
                    print("CONNECT WORKER:", ptr)
                    if ptr == 0:
                        print(f"PTR:{ptr}, NEW_COMMAND:{new_command}")
                    new_command = environ_command + new_command

                    # print(f"PTR:{ptr}, NEW_COMMAND:{new_command}")
                    # print(str(nodes))
                    pssh_pool.set_command(ptr, new_command, v['gpus'][i]['idx'])
                    ptr += 1
                    base_available_device_ids.add(v['gpus'][i]['idx'])

            print(f"node_info:{node_info}")
            print(f"gpu_info:{es.available_gpu_info()}")
            print(f"BASE_AVAILABLE_DEVICES:{base_available_device_ids}")

            
        print("Remain lives:", recover)

        # shared arrs for multiprocessing communication
        logging.basicConfig()
        arrs = {}
        arrs['arr'] = multiprocessing.Array("i", [0], lock=True)
        arrs['exit_arr'] = multiprocessing.Array("i", [0] * MAX_GPUS, lock=True)
        arrs['error_arr'] = multiprocessing.Array("i", [0] * MAX_GPUS, lock=True)
        arrs['worker_terminal'] = multiprocessing.Array("i", [0] * MAX_GPUS, lock=True)
        arrs['last_heartbeat']= multiprocessing.Array("d", [0.0] * MAX_GPUS, lock=True)
        arrs['stop'] = multiprocessing.Array("i", [0], lock=True)
        arrs['device_to_rank'] = multiprocessing.Array("i", [-1] * MAX_GPUS, lock=True)
        arrs['host_name_to_idx'] = {}
        arrs['host_name_to_local_idx'] = {}
        arrs['disabled_num_ranks'] = set()

        #initialize device to rank, we need a strategy for this
        if strategy is not None:
            # print(type(strategy['rank_to_device_mapping']), strategy['rank_to_device_mapping'])
            # print(eval(strategy['rank_to_device_mapping']))
            r2d = eval(strategy['rank_to_device_mapping'])
            print(f"rank2device:{r2d}")
            for k, v in r2d.items():
                arrs['device_to_rank'][v] = k 
            # for i in range(MAX_WORKERS):
            #     print(f"device2rank, idx={i}, val={arrs['device_to_rank'][i]}")
            last_index = -1
            for k, v in es.available_gpu_info().items():
                node_to_device_ids = []
                node_to_local_device_ids = []
                for gpu_item in v['gpus']:
                    node_to_device_ids.append(gpu_item['idx'])
                    node_to_local_device_ids.append(gpu_item['local_idx'])
                    if (gpu_item['idx'] != last_index + 1):
                        for i in range(last_index + 1, gpu_item['idx']):
                            arrs['disabled_num_ranks'].add(i)
                    last_index = gpu_item['idx']
                # check if the dead devices finally
                arrs['host_name_to_idx'][v['hostname']] = node_to_device_ids.copy()
                arrs['host_name_to_local_idx'][v['hostname']] = node_to_local_device_ids.copy()

            # check if the dead devices finally
            last_rank = len(es.available_gpu_info().items()) * DEVICES_PER_NODE - 1
            if last_index != last_rank:
                for i in range(last_index + 1, last_rank + 1):
                    arrs['disabled_num_ranks'].add(i)

            print(f"arrs['host_name_to_idx']:{arrs['host_name_to_idx']}")
            print(f"arrs['host_name_to_local_idx']:{arrs['host_name_to_local_idx']}")
            print(f"arrs['disabled_num_ranks']:{arrs['disabled_num_ranks']}")
            
        
        # p = multiprocessing.Process(target=serve, args=(arr, exit_arr, last_heartbeat, pssh_pool, port))
        p = multiprocessing.Process(target=serve, args=(arrs, pssh_pool, port))
        p.start()
        shared_dict = {}
        # TODO node detect thread, if ngpus changed(add), we should restart our training 
        node_detect_proc = threading.Thread(target=detect_gpu_nums, args=(nodes, args, shared_dict))
        node_detect_proc.start()

        # elastic restart
        if (recover <= ori_recover):
            for i in range(pssh_pool.nworkers()):
                pssh_pool.run_command(i)
        server_status = SERVER_STATUS.EXIT
        while (arrs['arr'][0] == 0 or arrs['arr'][0] > sum(arrs['exit_arr'])) and (sum(arrs['error_arr']) == 0):
            if (not node_detect_proc.is_alive()):
                cur_gpu_num = shared_dict['ngpu']
                cur_device_ids = shared_dict['available_device_ids']
                for dev_id in cur_device_ids:
                    if dev_id not in base_available_device_ids:
                        print(f"find new available device id:{dev_id}, "
                              f"base_available_device_ids:{base_available_device_ids}")
                        # server_status = SERVER_STATUS.RESTART
                        # break
                node_detect_proc = threading.Thread(target=detect_gpu_nums, args=(nodes, args, shared_dict))
                # node_detect_proc.start()
            time.sleep(5)
            cur_time = time.time()
            for i in range(arrs['arr'][0]):
                if arrs['last_heartbeat'][i] > 0:
                    interval = cur_time - arrs['last_heartbeat'][i]
                else:
                    interval = 0
                if (interval > 10):
                    arrs['exit_arr'][i] = 1
                    arrs['error_arr'][i] = 1
            #     print("Interval of Rank ", i, ":", interval, arrs['exit_arr'][i])
            # print("PSSH_POOL_NUM:", pssh_pool.nworkers())
            # print("Arr0:", arrs['arr'][0], "SumExit:", sum(arrs['exit_arr']))

        if server_status == SERVER_STATUS.RESTART:
            arrs['stop'][0] = 1    


        # end of server
        if (args.mpi):
            while(sum(arrs['exit_arr']) + sum(arrs['error_arr']) < arrs['arr'][0]):
                cur_time = time.time()
                for i in range(arrs['arr'][0]):
                    if arrs['last_heartbeat'][i] > 0:
                        interval = cur_time - arrs['last_heartbeat'][i]
                    else:
                        interval = 0
                    if (interval > 10):
                        arrs['exit_arr'][i] = 1
                        arrs['error_arr'][i] = 1
                    print("Interval of Rank ", i, ":", interval, arrs['exit_arr'][i])
                if (sum(arrs['error_arr']) == 0):
                # only for test
                    time.sleep(1)
                else:
                    print("Terminal:", sum(arrs['exit_arr']), "Error:", sum(arrs['error_arr']))
                    time.sleep(1)
        else:
            while(sum(arrs['worker_terminal']) + sum(arrs['error_arr']) < arrs['arr'][0]):
                cur_time = time.time()
                for i in range(arrs['arr'][0]):
                    if arrs['last_heartbeat'][i] > 0:
                        interval = cur_time - arrs['last_heartbeat'][i]
                    else:
                        interval = 0
                    if (interval > 10):
                        arrs['exit_arr'][i] = 1
                        arrs['error_arr'][i] = 1
                    # print("Interval of Rank ", i, ":", interval, arrs['exit_arr'][i])
                if (sum(arrs['error_arr']) == 0):
                # only for test
                    # if sum(arrs['exit_arr']) == arrs['arr'][0]:
                    #     break
                    time.sleep(1)
                else:
                    server_status = SERVER_STATUS.ERROR
                    arrs['stop'][0] = 1
                    print("Terminal:", sum(arrs['worker_terminal']), "Error:", sum(arrs['error_arr']))
                    time.sleep(1)
                print("Shutdown nums:", sum(arrs['worker_terminal']) + sum(arrs['error_arr']), 
                      "Exit nums:", sum(arrs['exit_arr']),
                      "Total nums:", arrs['arr'][0])
                    
        print(f"SERVER_STAUS:{server_status}")
        if sum(arrs['error_arr']) == 0:
            # end training.
            # this means our training come to a end without any error. Congratulations!
            print("Server Stopped.")
            # recover = 0
            p.terminate()   
        else:
            # training interruptted by some unknown reasons(Mostly a node or a GPU died). 
            # We need to recover our training. 
            print("Server Killed. Terminal:", sum(arrs['worker_terminal']), "Error:", sum(arrs['error_arr']))
            p.terminate()     
        recover -= 1
        print(f"Recover:{recover}")