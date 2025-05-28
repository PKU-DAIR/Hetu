from concurrent import futures
from kv_store import key_value_store_server
import logging
import multiprocessing.spawn
import grpc
import threading
import time
import multiprocessing
import heturpc_pb2
import heturpc_pb2_grpc
from kv_store.server import key_value_store_server

_MAX_UNFOUND_TIMES = 10000
_MAX_GRPC_WORKERS = 64

@key_value_store_server
class DeviceController(heturpc_pb2_grpc.DeviceControllerServicer):
    def __init__(self, arr, exit_arr, last_heartbeat) -> None:
        super().__init__()
        self.arr = arr
        self.exit_arr = exit_arr
        self.last_heartbeat = last_heartbeat
        # Conditions
        self.lock = threading.Lock()
        self.double_cond = threading.Condition()
        self.int_cond = threading.Condition()
        self.string_cond = threading.Condition()
        self.bytes_cond = threading.Condition()
        self.json_cond = threading.Condition()
        self.barrier_cond = threading.Condition()
        self.nccl_cond = threading.Condition()
        self.hostname_cond = threading.Condition()
        self.deviceinfo_cond = threading.Condition()
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
        # dicts
        self.double_dict = {}
        self.int_dict = {}
        self.string_dict = {}
        self.bytes_dict = {}
        self.json_dict = {}

    def Connect(self, request, context):
        with self.lock:
            nodename = "node-" + request.hostname
            if nodename not in self.local_worldsizes:
                self.local_worldsizes[nodename] = 0
                self.nodenames.append(nodename)
            print(nodename, "connect")
            self.local_worldsizes[nodename] += 1
            self.worldsize += 1
        return heturpc_pb2.ConnectReply(status=1)

    def GetRank(self, request, context):
        local_rank = 0
        with self.lock:
            nodename = "node-" + request.name
            if nodename not in self.local_ranks:
                self.local_ranks[nodename] = 0
            for nodename_ in self.nodenames:
                if nodename_ == nodename:
                    local_rank += self.local_ranks[nodename]
                    self.local_ranks[nodename] += 1
                    break
                else:
                    local_rank += self.local_worldsizes[nodename_]
            print(request.name, "rank", local_rank, "confirm")
            self.last_heartbeat[int(local_rank)] = time.time()
            self.arr[0] += 1
        return heturpc_pb2.RankReply(rank=local_rank)

    def CommitHostName(self, request, context):
        with self.hostname_cond:
            self.num_hostnames += 1
            self.hostnames[request.rank] = request.hostname
            self.hostname_cond.notify_all()
        return heturpc_pb2.CommitHostNameReply(status=1)

    def GetHostName(self, request, context):
        with self.hostname_cond:
            while request.rank not in self.hostnames:
                self.hostname_cond.wait()
            hostname = self.hostnames[request.rank]
        return heturpc_pb2.GetHostNameReply(hostname=hostname)

    def CommitDeviceInfo(self, request, context):
        with self.deviceinfo_cond:
            self.num_deviceinfos += 1
            self.deviceinfos[request.rank] = (request.type, request.index, request.multiplex)
            self.deviceinfo_cond.notify_all()
        return heturpc_pb2.CommitDeviceInfoReply(status=1)

    def GetDeviceInfo(self, request, context):
        with self.deviceinfo_cond:
            while request.rank not in self.deviceinfos:
                self.deviceinfo_cond.wait()
            mtype, mindex, mmultiplex = self.deviceinfos[request.rank]
        return heturpc_pb2.GetDeviceInfoReply(type=mtype, index=mindex, multiplex=mmultiplex)

    def CommitNcclId(self, request, context):
        with self.nccl_cond:
            world_rank = tuple(request.world_rank) + (int(request.stream_id),)
            self.nccl_ids[world_rank] = request.nccl_id
            self.nccl_cond.notify_all()
        return heturpc_pb2.CommitNcclIdReply(status=1)

    def GetNcclId(self, request, context):
        world_rank = tuple(request.world_rank) + (int(request.stream_id),)
        with self.nccl_cond:
            while world_rank not in self.nccl_ids:
                self.nccl_cond.wait()
            nccl_id = self.nccl_ids[world_rank]
        return heturpc_pb2.GetNcclIdReply(nccl_id=nccl_id)

    def Exit(self, request, context):
        with self.lock:
            self.exit_arr[request.rank] = 1
            self.exit_nums += 1
        print(self.exit_nums, "of", self.worldsize, "ranks have exited")
        return heturpc_pb2.ExitReply(status=1)

    def PutDouble(self, request, context):
        with self.double_cond:
            self.double_dict[request.key] = request.value
            self.double_cond.notify_all()
        return heturpc_pb2.PutDoubleReply(status=1)

    def GetDouble(self, request, context):
        with self.double_cond:
            while request.key not in self.double_dict:
                self.double_cond.wait()
            value = self.double_dict[request.key]
        return heturpc_pb2.GetDoubleReply(value=value)

    def RemoveDouble(self, request, context):
        with self.double_cond:
            unfound_times = 0
            while request.key not in self.double_dict and unfound_times < _MAX_UNFOUND_TIMES:
                self.double_cond.wait(timeout=0.0001)
                unfound_times += 1
            if unfound_times == _MAX_UNFOUND_TIMES:
                return heturpc_pb2.RemoveDoubleReply(message="not found:" + request.key)
            else:
                self.double_dict.pop(request.key)
        return heturpc_pb2.RemoveDoubleReply(message="already remove:" + request.key)

    def PutInt(self, request, context):
        with self.int_cond:
            self.int_dict[request.key] = request.value
            self.int_cond.notify_all()
        return heturpc_pb2.PutIntReply(status=1)

    def GetInt(self, request, context):
        with self.int_cond:
            while request.key not in self.int_dict:
                self.int_cond.wait()
            value = self.int_dict[request.key]
        return heturpc_pb2.GetIntReply(value=value)

    def RemoveInt(self, request, context):
        with self.int_cond:
            unfound_times = 0
            while request.key not in self.int_dict and unfound_times < _MAX_UNFOUND_TIMES:
                self.int_cond.wait(timeout=0.0001)
                unfound_times += 1
            if unfound_times == _MAX_UNFOUND_TIMES:
                return heturpc_pb2.RemoveIntReply(message="not found:" + request.key)
            else:
                self.int_dict.pop(request.key)
        return heturpc_pb2.RemoveIntReply(message="already remove:" + request.key)

    def PutString(self, request, context):
        with self.string_cond:
            self.string_dict[request.key] = request.value
            self.string_cond.notify_all()
        return heturpc_pb2.PutStringReply(status=1)

    def GetString(self, request, context):
        with self.string_cond:
            while request.key not in self.string_dict:
                self.string_cond.wait()
            value = self.string_dict[request.key]
        return heturpc_pb2.GetStringReply(value=value)

    def RemoveString(self, request, context):
        with self.string_cond:
            unfound_times = 0
            while request.key not in self.string_dict and unfound_times < _MAX_UNFOUND_TIMES:
                self.string_cond.wait(timeout=0.0001)
                unfound_times += 1
            if unfound_times == _MAX_UNFOUND_TIMES:
                return heturpc_pb2.RemoveStringReply(message="not found:" + request.key)
            else:
                self.string_dict.pop(request.key)
        return heturpc_pb2.RemoveStringReply(message="already remove:" + request.key)

    def PutBytes(self, request, context):
        with self.bytes_cond:
            self.bytes_dict[request.key] = request.value
            self.bytes_cond.notify_all()
        return heturpc_pb2.PutBytesReply(status=1)

    def GetBytes(self, request, context):
        with self.bytes_cond:
            while request.key not in self.bytes_dict:
                self.bytes_cond.wait()
            value = self.bytes_dict[request.key]
        return heturpc_pb2.GetBytesReply(value=value)

    def RemoveBytes(self, request, context):
        with self.bytes_cond:
            unfound_times = 0
            while request.key not in self.bytes_dict and unfound_times < _MAX_UNFOUND_TIMES:
                self.bytes_cond.wait(timeout=0.0001)
                unfound_times += 1
            if unfound_times == _MAX_UNFOUND_TIMES:
                return heturpc_pb2.RemoveBytesReply(message="not found:" + request.key)
            else:
                self.bytes_dict.pop(request.key)
        return heturpc_pb2.RemoveBytesReply(message="already remove:" + request.key)

    def PutJson(self, request, context):
        with self.json_cond:
            self.json_dict[request.key] = request.value
            import json
            my_json = json.loads(request.value)
            # print("ReCEIVEJSON:", my_json)
            self.json_cond.notify_all()
        return heturpc_pb2.PutJsonReply(status=1)

    def GetJson(self, request, context):
        with self.json_cond:
            while request.key not in self.json_dict:
                self.json_cond.wait()
            value = self.json_dict[request.key]
        return heturpc_pb2.GetJsonReply(value=value)

    def RemoveJson(self, request, context):
        with self.json_cond:
            unfound_times = 0
            while request.key not in self.json_dict and unfound_times < _MAX_UNFOUND_TIMES:
                self.json_cond.wait(timeout=0.0001)
                unfound_times += 1
            if unfound_times == _MAX_UNFOUND_TIMES:
                return heturpc_pb2.RemoveJsonReply(message="not found:" + request.key)
            else:
                self.json_dict.pop(request.key)
        return heturpc_pb2.RemoveJsonReply(message="already remove:" + request.key)

    def Barrier(self, request, context):
        world_rank = tuple(request.world_rank)
        with self.barrier_cond:
            if world_rank not in self.barrier_nums:
                self.barrier_nums[world_rank] = 0
            if request.rank == world_rank[0]:
                self.barrier_end_nums[world_rank] = 0
            self.barrier_nums[world_rank] += 1
            self.barrier_cond.notify_all()
            while self.barrier_nums[world_rank] < len(world_rank):
                self.barrier_cond.wait()
            self.barrier_end_nums[world_rank] += 1
            self.barrier_cond.notify_all()
            while self.barrier_end_nums[world_rank] < len(world_rank):
                self.barrier_cond.wait()
            if request.rank == world_rank[0]:
                self.barrier_nums[world_rank] = 0
        return heturpc_pb2.BarrierReply(status=1)

    def HeartBeat(self, request, context):
        with self.lock:
            self.last_heartbeat[request.rank] = time.time()
        return heturpc_pb2.HeartBeatReply(status=1)

def serve(arr, exit_arr, last_heartbeat, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=_MAX_GRPC_WORKERS))
    heturpc_pb2_grpc.add_DeviceControllerServicer_to_server(DeviceController(arr, exit_arr, last_heartbeat), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()
    print("Server terminated")

def server_launch(port):
    logging.basicConfig()
    arr = multiprocessing.Array("i", [0], lock=True)
    exit_arr = multiprocessing.Array("i", [0] * _MAX_GRPC_WORKERS, lock=True)
    last_heartbeat = multiprocessing.Array("d", [0.0] * _MAX_GRPC_WORKERS, lock=True)
    p = multiprocessing.Process(target=serve, args=(arr, exit_arr, last_heartbeat, port))
    p.start()
    while True:
        time.sleep(5)
        cur_time = time.time()
        for i in range(arr[0]):
            interval = cur_time - last_heartbeat[i]
            if (interval > 10):
                exit_arr[i] = 1
            print("Heartbeat interval of rank", i, "is", interval)
        if arr[0] != 0 and arr[0] <= sum(exit_arr):
            break
    print("Server Stopped.")
    p.terminate()
