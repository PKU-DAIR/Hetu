import os
import yaml
import hydra
from omegaconf import OmegaConf
import time
from pssh.clients import ParallelSSHClient
from pssh.utils import enable_host_logger
# from heturpc_polling_server import server_launch
from heturpc_async_server import server_launch
import multiprocessing.spawn

# enable_host_logger()

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def pssh(config):
    hostnames = []
    ports = []
    passwords = []
    if config.hosts is None:
        hostnames = ['localhost'] * config.num_gpus
        ports = [22] * config.num_gpus
        passwords = [None] * config.num_gpus
    else:
        host_info = read_yaml(config.hosts)
        max_restart_times = host_info['max_restart_times']
        heartbeat_interval = float(host_info['heartbeat_interval'])
        for host in host_info['hosts']:
            print(host)
            addr = str(host['addr'])
            port = int(host['port']) if 'port' in host else 22
            password = str(host['password']) if 'password' in host else None
            initial_workers = int(host['initial_workers'])
            min_workers = int(host['min_workers'])
            max_workers = int(host['max_workers'])
            for i in range(initial_workers):
                # workaround: 当不使用host的全部gpu时优先按顺序使用
                if len(hostnames) == config.num_gpus:
                    break
                hostnames.append(addr)
                ports.append(port)
                passwords.append(password)
    print("HostNames:", hostnames)
    train_command = config.command
    cwd = os.getcwd()
    cmd = "cd " + cwd 
    cmd += f" && source {config.envs} && " 
    cmd_list = []
    for i, hostname in enumerate(hostnames):
        # 请注意log编号目前并不等于rank编号
        # log编号是进程编号
        # 但不能保证分配到同样编号的rank
        if config.nsys_profile and i == 0:
            new_train_command = f"nsys profile -o {config.log_path}/report_{i} " + train_command
        else:
            new_train_command = train_command
        cmd_list.append(cmd + f"export HETU_LOCAL_HOSTNAME={hostname} && " + new_train_command + f" 2>&1 | tee {config.log_path}" + "/log_" + f"{i}" + ".log")
    clients = []
    outputs = []
    for hostname, port, password, cmd in zip(hostnames, ports, passwords, cmd_list):
        client = ParallelSSHClient([hostname], port=port, password=password)
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

@hydra.main(config_path=None, config_name='config', version_base=None)
def main(config):
    config = OmegaConf.select(config, "rpc")
    os.makedirs(config.log_path, exist_ok=True)
    p = multiprocessing.Process(target=server_launch, args=(str(config.server_port),))
    p.start()
    # workaround: clients may start earlier than the server
    # need to use retry-until-connect approach
    # now simply wait 3 seconds to ensure the server is lauched before clients
    time.sleep(3)
    pssh(config)
    p.join()

if __name__ == '__main__':
    main()
