import os
import argparse
import yaml
# from heturpc_polling_server import server_launch
import multiprocessing.spawn
import threading
import time
import multiprocessing
import subprocess
import grpc
import heturpc_pb2
import heturpc_pb2_grpc
import paramiko
import logging
logging.getLogger("paramiko").setLevel(logging.WARNING)
import re
import atexit
import signal
import sys

from enum import Enum
class PSSHStatus(Enum):
    success = 0
    failed = 1

# enable_host_logger()
class PSSHHandler:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.device_idx = 0
        self.pssh_client = None
        self.status = PSSHStatus.success
        self.process = None
        self.command = ""
        self.run_log = ""
        self.error_log = ""
    
    def __del__(self):
        if self.pssh_client is not None:
            self.pssh_client.close()
    
    def device_index(self):
        return self.device_idx
    
    def register_client(self, hostnames, port):
        # self.pssh_client = ParallelSSHClient(hostnames, port=port)
        if self.pssh_client is not None:
            self.pssh_client.close()
        self.pssh_client = paramiko.SSHClient()
        self.pssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
        self.pssh_client.connect(hostnames[0], port=port)

    def run_command(self, cmd = None):
        if cmd == None:
            cmd = self.command

        pattern = r'log_\d+\.txt'
        replacement = f"log_{str(self.worker_id)}.txt"

        cmd = re.sub(pattern, replacement, cmd)
        # if self.worker_id == 0:
        #     print("RUN COMMAND:", cmd + " --pid " + str(self.worker_id))
        self.command = cmd
        # self.run_log = self.pssh_client.run_command(cmd)
        _, self.run_log, self.error_log = \
        self.pssh_client.exec_command(cmd + " --pid " + str(self.worker_id))
        # self.pssh_client.join()
    
    def set_command(self, cmd, device_idx):
        self.command = cmd
        self.device_idx = device_idx
    
    def get_command(self):
        return self.command
    
    def get_run_log(self):
        self.pssh_client.join()
        for line in self.run_log.stdout:
            print(line)

    
    def run(self, cmd):
        self.process = multiprocessing.Process(target=self.run_command, args=(cmd,))
        self.process.start()

    def terminal(self, status:PSSHStatus):
        if status == PSSHStatus.success:
            self.process.join()
        elif status == PSSHStatus.failed:
            self.process.terminate()

class PSSHHandlerPool:
    def __init__(self):
        self.worker_list = []
        self.num_workers = 0
        
    def alloc_workers(self, num_workers=32):
        if (self.num_workers < num_workers):
            for i in range(self.num_workers, num_workers):
                self.worker_list.append(PSSHHandler(i))
        self.num_workers = num_workers
        print("CUR_WORKERS:", self.num_workers)
        devices = []
        for i in range(self.num_workers):
            devices.append(self.worker_list[i].device_index())
        # print("DEVICES:", devices)
        # return self.worker_list
    
    def register_client(self, idx, hostnames, port):
        self.worker_list[idx].register_client(hostnames, port)
    
    def run(self, idx, cmd):
        self.worker_list[idx].run(cmd)

    def set_command(self, idx, cmd, device_idx):
        self.worker_list[idx].set_command(cmd, device_idx)

    def get_command(self, idx):
        return self.worker_list[idx].get_command()
    
    def run_command(self, idx, cmd = None):
        self.worker_list[idx].run_command(cmd)
    
    def get_run_log(self, idx):
        return self.worker_list[idx].get_run_log()
    
    def nworkers(self):
        return self.num_workers

class PSSHWorker:
    def __init__(self, id=0):
        self.id = id
        self.cmd = ""
        self.main_process = None
        self.log_file = None
    
    def run_command(self, cmd):
        print("WORKER:", cmd)
        result = subprocess.run([cmd], capture_output=True, text=True, shell=True, check=True)
        # raise ValueError("Non-zero exitcode:" + str(result.returncode))
        # if result.returncode != 0:
        #     raise ValueError("Non-zero exitcode:" + str(result.returncode))

    def run_command_async(self, subprocess):
        stdout, stderror = subprocess.communicate()
        # result = subprocess.communicate()
        # raise ValueError("Non-zero exitcode:" + str(result.returncode))
        # if result.returncode != 0:
        #     raise ValueError("Non-zero exitcode:" + str(result.returncode))

    
    def run(self, args):
        # self.process = multiprocessing.Process(target=self.run_command, args=(cmd,))
        # self.process.start()
        cmd = args.command
        pid = args.pid
        recover = 1

        tmp_dir = "./tmp_logs"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        log_name = "log_" + str(pid) + ".txt"
        wlog_name = "wlog_" + str(pid) + ".txt"
        tmp_log = os.path.join(tmp_dir, log_name)
        tmp_wlog = os.path.join(tmp_dir, wlog_name)
        self.log_file = open(tmp_log, "w")

        f = open(tmp_wlog, 'w')
        def printf(file, txt):
            file.write(txt + "\n")
            file.flush()

        while recover:
            server_path = args.server_addr + ":" + args.server_port
            ready_for_connect = False
            st_time = time.time()
            remain_failed = 10
            printf(f, f"server path:{server_path}")
            while (not ready_for_connect) and (remain_failed > 0):
                test_channel = grpc.insecure_channel(server_path)
                test_stub = heturpc_pb2_grpc.DeviceControllerStub(test_channel)
                try:
                    response = test_stub.PutInt(heturpc_pb2.PutIntRequest(key=str(pid), value=0))
                    ready_for_connect = True
                    printf(f, f"Pid {pid} Connection is successful.")
                except grpc.RpcError as e:
                    printf(f, f"Pid {pid} Connection is unsuccessful.")
                time.sleep(1)
                remain_failed -= 1
            with grpc.insecure_channel(server_path) as channel:
                stub = heturpc_pb2_grpc.DeviceControllerStub(channel)
                # self.process = multiprocessing.Process(target=self.run_command, args=(cmd,))
                # self.process = self.run_command(cmd)

                self.main_process = subprocess.Popen([cmd], shell=True, stdout=self.log_file,
                                                     stderr=self.log_file,
                                                     preexec_fn=os.setsid)
                pgid = os.getpgid(self.main_process.pid)
                printf(f, f"cur_pid:{pgid}")
                def cleanup():
                    if self.main_process.poll() is None:
                        os.killpg(pgid, signal.SIGTERM)
                    else:
                        os.killpg(pgid, signal.SIGTERM)
                    if self.log_file is not None:
                        self.log_file.close()

                atexit.register(cleanup)
                def signal_handler(signum, frame):
                    cleanup()
                    printf(f, "killing script")
                    sys.exit(1)

                # signal handler, if worker killed, the training thread should be killed. 
                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)
                response = stub.PutString(heturpc_pb2.PutStringRequest(key="PROCESS", 
                                                                       value=str(self.main_process.pid)))
                self.process = threading.Thread(target=self.run_command_async, 
                                                args=(self.main_process,))
                self.process.start()
                printf(f, f"command begin")
                while True:
                    # response = stub.HeartBeat(heturpc_pb2.HeartBeatRequest(rank=pid))
                    if self.main_process.poll() is not None:
                        # response = stub.AlreadyStop(heturpc_pb2.AlreadyStopRequest(rank=pid))
                        # os.killpg(pgid, signal.SIGTERM)
                        response = stub.WorkerStop(heturpc_pb2.WorkerStopRequest(rank=pid))
                        response = stub.PutString(heturpc_pb2.PutStringRequest(key="DEAD", value=str(pid)))
                        break
                    else:
                        # self.process.join(timeout=5)
                        # response = stub.HeartBeat(heturpc_pb2.HeartBeatRequest(rank=pid))
                        try:
                            response = stub.PutString(heturpc_pb2.PutStringRequest(key="ALIVE", value=str(pid)))
                            printf(f, f"alive")
                            time.sleep(5)
                            response = stub.AlreadyStop(heturpc_pb2.AlreadyStopRequest(rank=pid))
                            if (response.status == 1):
                                response = stub.WorkerStop(heturpc_pb2.WorkerStopRequest(rank=pid))
                                response = stub.PutString(heturpc_pb2.PutStringRequest(key="KILLED", value=str(pid)))
                                os.killpg(pgid, signal.SIGTERM)
                                break

                        except:
                            printf(f, f"error")
                            # response = stub.PutString(heturpc_pb2.PutStringRequest(key="ERROR", value=str(pid)))
                            os.killpg(pgid, signal.SIGTERM)
                            printf(f, f"kill")
                            break
            # response = stub.PutString(heturpc_pb2.PutStringRequest(key="WORKER END", 
            #                                                        value=str(pid)))
            recover -= 1
        if self.log_file is not None:
            self.log_file.close()
                
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
    )
    parser.add_argument(
        "--command", type=str, default='uname', help="command for pssh"
    )
    parser.add_argument(
        "--pid", type=int, default=0, help="process id of worker"
    )
    parser.add_argument(
        "--recover", type=int, default=1, help="recover times in elastic training"
    )
    args = parser.parse_args()
    worker = PSSHWorker()
    worker.run(args)
