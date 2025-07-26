import os
import argparse
# from heturpc_polling_server import server_launch
from heturpc_elastic_server import server_launch
import multiprocessing.spawn
from pssh_workers import *
from multiprocessing.managers import BaseManager
from elastic_arg_parser import ElasticStrategy, \
                               replace_argument, get_argument, add_argument, delete_argument
from utils import read_yaml

class HetuManager(BaseManager):
    pass

HetuManager.register("PSSHPool", PSSHHandlerPool)

# enable_host_logger()

def pssh(args, pssh_pool: PSSHHandlerPool):
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
    worker_path = os.environ['HETU_HOME'] + "/python/hetu/rpc/pssh_workers.py"
    cwd = os.getcwd()
    cmd = "cd " + cwd 
    cmd += f" && source {args.envs} && " + "python3 " + worker_path + \
           " --server_addr " + args.server_addr + \
           " --server_port " + args.server_port + \
           " --command \"" + train_command
    print(cmd)
    cmd_list = []
    for i in range(len(hostnames)):
        # 请注意log编号目前并不等于rank编号
        # log编号是进程编号
        # 但不能保证分配到同样编号的rank
        # cmd_list.append(cmd + "\"")
        cmd_list.append(cmd + f" 2>&1 | tee {args.log_path}" + "/log_" + f"{i}" + ".txt" + "\"")
    clients = []
    outputs = []
    ptr = 0
    pssh_pool.alloc_workers(len(hostnames))
    if args.mpi:
        pssh_pool.register_client(ptr, ['localhost'], 22)
        print("CMDD:", cmd_list[0])
        pssh_pool.run_command(ptr, cmd_list[0])
    else:     
        for hostname, cmd in zip(hostnames, cmd_list):
            
            pssh_pool.register_client(ptr, [hostname], 22)

            pssh_pool.run_command(ptr, cmd)
            # outputs.append(output)
            ptr += 1

    for client in clients:
        client.join() 
    for output in outputs:
        for host_out in output:
            for line in host_out.stdout:
                print(line)
            # for line in host_out.stderr:
            #     print("[stderr]:", line)
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
    parser.add_argument(
        "--mpi", type=bool, default=False, help="whether or not use mpi launch"
    )
    parser.add_argument(
        "--ssh_port", type=str, default='22', help="ssh port"
    )
    parser.add_argument(
        "--recover", type=int, default=1, help="max recover times"
    )
    args = parser.parse_args()
    print("ARGS:\n", args)
    with HetuManager() as manager:
        pssh_pool = manager.PSSHPool()
        p = multiprocessing.Process(target=server_launch, args=(args.server_port, pssh_pool, args))
        p.start()
        # pssh(args, pssh_pool)
        p.join()