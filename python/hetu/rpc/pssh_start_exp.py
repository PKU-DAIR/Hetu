import os
import time
import argparse
import yaml
import threading
from pssh.clients import ParallelSSHClient
from pssh.exceptions import Timeout
from pssh.utils import enable_host_logger
# from heturpc_polling_server import server_launch
from heturpc_async_server_exp import server_launch
import multiprocessing.spawn

# enable_host_logger()
TIMEOUT = 180 # 如果180s之间log都没有增加我们认为其卡死

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def pssh(args):
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
                if len(hostnames) == args.ngpus:
                    break
                hostnames.append(addr)
    print("HostNames:", hostnames)
    train_command = args.command
    cwd = os.getcwd()
    cmd = "cd " + cwd 
    cmd += f" && source {args.envs} && " 
    cmd_list = []
    log_list = []
    for i, hostname in enumerate(hostnames):
        # 请注意log编号目前并不等于rank编号
        # log编号是进程编号
        # 但不能保证分配到同样编号的rank
        log_path = args.log_path + "/log_" + f"{i}" + ".log"
        log = open(log_path, 'w')
        log.close()
        log_list.append(log_path)
        cmd_list.append(cmd + f"export HETU_LOCAL_HOSTNAME={hostname} && " + train_command + f" 2>&1 | tee {log_path}")
    clients = []
    outputs = []
    for hostname, cmd in zip(hostnames, cmd_list):
        client = ParallelSSHClient([hostname], user='root', port=36000)
        output = client.run_command(cmd, use_pty=True)
        clients.append(client)
        outputs.append(output)
    clients_terminate = {}
    def monitor_client(lock, i, client, output):
        nonlocal log_list, clients_terminate
        cnt = 0
        log_size = os.path.getsize(log_list[i])
        while True:
            try:
                client.join(timeout=0.1)
            except Timeout:
                pass
            else:
                print(f"Client {i} (unrelated to rank) is finished, terminate it")
                break
            time.sleep(1)
            cnt += 1
            if cnt == TIMEOUT:
                new_size = os.path.getsize(log_list[i])
                if new_size == log_size:
                    print(f"Client {i} (unrelated to rank) is timeout, terminate it")
                    break
                # print(f"Client {i} (unrelated to rank): set new log size {new_size}")
                log_size = new_size
                cnt = 0
        with lock:
            clients_terminate[client] = True
    threads = []
    lock = threading.Lock()
    for i, (client, output) in enumerate(zip(clients, outputs)):
        thread = threading.Thread(target=monitor_client, args=(lock, i, client, output))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    for i, (client, output) in enumerate(zip(clients, outputs)):
        # Note: closing channel which has PTY has the effect of terminating
        # any running processes started on that channel.
        assert clients_terminate[client] == True, f"Client {client} should already be terminated"
        for host_out in output:
            print(f"Client {i} (unrelated to rank) closing")
            host_out.client.close_channel(host_out.channel)
    print("All clients are closed, terminate the server")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command", type=str, default='uname', help="command for pssh"
    )
    parser.add_argument(
        "--ngpus", type=int, default=8, help="num gpus"
    )
    parser.add_argument(
        "--hosts", type=str, help="multi-node hosts"
    )
    parser.add_argument(
        "--envs", type=str, help="multi-node shared envs"
    )
    parser.add_argument(
        "--log_path", type=str, help="log folder path"
    )
    args = parser.parse_args()
    # 一些卡死的情况只有client端能够知晓而heartbeat仍然是正常的
    # 此时就需要进程间额外通信
    message_queue = multiprocessing.Queue() 
    p = multiprocessing.Process(target=server_launch, args=(args.server_port, message_queue))
    p.start()
    pssh(args)
    # currently heartbeat may exist but training may get stuck
    # we need to inform the server manually if clients are all terminated
    message_queue.put("terminate")
    p.join()
