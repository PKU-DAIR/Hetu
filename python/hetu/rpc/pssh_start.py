import os
import argparse
import yaml
from pssh.clients import ParallelSSHClient
from pssh.utils import enable_host_logger
# from heturpc_polling_server import server_launch
from heturpc_async_server import server_launch
import multiprocessing.spawn

# enable_host_logger()

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
                hostnames.append(addr)
    print("HostNames:", hostnames)
    train_command = args.command
    cwd = os.getcwd()
    cmd = "cd " + cwd 
    cmd += f" && source {args.envs} && " + train_command 
    print(cmd)
    cmd_list = []
    for i in range(len(hostnames)):
        # 请注意log编号目前并不等于rank编号
        # log编号是进程编号
        # 但不能保证分配到同样编号的rank
        cmd_list.append(cmd + f" 2>&1 | tee {args.log_path}" + "/log_" + f"{i}" + ".txt")
    clients = []
    outputs = []
    for hostname, cmd in zip(hostnames, cmd_list):
        client = ParallelSSHClient([hostname])
        # If password is needed, you should ssh like this
        # client = ParallelSSHClient([hostname], port=args.pssh_port, password=args.pssh_password)
        output = client.run_command(cmd)
        clients.append(client)
        outputs.append(output)
    for client in clients:
        client.join() 
    for output in outputs:
        for host_out in output:
            for line in host_out.stderr:
                print("[stderr]:", line)
            '''
            for line in host_out.stdout:
                print(line)
            exit_code = host_out.exit_code
            '''
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command", type=str, default='uname', help="command for pssh"
    )
    parser.add_argument(
        "--pssh_port", type=str, default='60001', help="pssh's port"
    )
    parser.add_argument(
        "--pssh_password", type=str, default='xxx', help="pssh's password"
    )
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
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
    p = multiprocessing.Process(target=server_launch, args=(args.server_port,))
    p.start()
    pssh(args)
    p.join()