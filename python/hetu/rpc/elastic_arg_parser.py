import csv
import shlex
import os
import subprocess
from hetu.engine.strategy_ampelos import *
from hetu.engine.utils import *
from io import StringIO
import sys
import re
import json
import paramiko
import logging
logging.getLogger("paramiko").setLevel(logging.WARNING)

DEVICES_PER_NODE = 8
TIMEOUT = 5.0
DS_CONFIG_PATH = "examples/ampelos/ds_parallel_config"

def delete_argument(command_string, arg_name):
    args_list = shlex.split(command_string)
    new_args_list = []
    i = 0
    while i < len(args_list):
        if args_list[i] == arg_name:
            if args_list[i + 1][:2] == "--":
                i += 1
            else:
                i += 2
        else:
            new_args_list.append(args_list[i])
            i += 1

    new_command_string = ' '.join(arg for arg in new_args_list)
    return new_command_string

def replace_argument(command_string, arg_name, new_value):
    args_list = shlex.split(command_string)
    if arg_name not in args_list:
        return add_argument(command_string, arg_name, new_value)
    new_args_list = []
    i = 0
    while i < len(args_list):
        if args_list[i] == arg_name:
            new_args_list.append(arg_name)
            new_args_list.append(str(new_value))
            i += 2  
        else:
            new_args_list.append(args_list[i])
            i += 1

    new_command_string = ' '.join(arg for arg in new_args_list)
    return new_command_string

def add_argument(command_string, arg_name, new_value=None):
    args_list = shlex.split(command_string)
    new_args_list = []
    i = 0
    while i < len(args_list):
        if args_list[i][:2] == "--":
            new_args_list.append(arg_name)
            if new_value is not None:
                new_args_list.append(str(new_value))
            break
        new_args_list.append(args_list[i])
        i += 1
    
    while i < len(args_list):
        new_args_list.append(args_list[i])
        i += 1

    new_command_string = ' '.join(arg for arg in new_args_list)
    return new_command_string

def get_argument(command_string, arg_name):
    args_list = shlex.split(command_string)
    new_args_list = []
    i = 0
    while i < len(args_list):
        if args_list[i] == arg_name:
            if args_list[i + 1][:2] == "--":
                return True
            else:
                return args_list[i + 1]
        else:
            i += 1

    return False

class ElasticStrategy:
    def __init__(self, cmd="", node_info=None):
        self.cmd = cmd
        self.train_cmd = get_argument(cmd, "--command")
        self.base_config = None
        self.node_info = node_info
        if cmd != "":
            self.set_default_values()
        self.available_gpus = None
        self.strategy_dict = None
        self.parallel_config = None
        self.trainer_ctx_config = None
        self.hetero = False
        self.strategy_model = None
        self.trainer_ctxs = None
        self.strategy_args = None
        self.homo_neglect_keys = ["cp_list", "hetero_layers", "hetero_stages",
                                  "unused_rank", "rank_to_device_mapping"]
    
    def get_arguments(self, cmd: str, args: list):
        config_dict = {}
        for arg in args:
            config_dict[arg] = get_argument(cmd, "--" + arg)
        return config_dict
    
    def set_default_values(self):
        args = ["ds_parallel_config", "num_hidden_layers", "ngpus", "global_batch_size"]
        base_parallel_config = self.get_arguments(self.train_cmd, args)
        para_config = str(base_parallel_config["ds_parallel_config"])
        #dcp
        pattern = r'dcp\d+_'
        match = re.search(pattern, para_config)
        base_parallel_config['dcp'] = int(match.group().replace("dcp","").replace("_",""))
        #tp
        pattern = r'tp\d+_'
        match = re.search(pattern, para_config)
        base_parallel_config['tp'] = int(match.group().replace("tp","").replace("_",""))
        #pp
        pattern = r'pp\d+.'
        match = re.search(pattern, para_config)
        base_parallel_config['pp'] = int(match.group().replace("pp","").replace(".",""))
        #zero
        ds_parallel_config = json.load(open(para_config, 'r'))
        base_parallel_config['zero'] = ds_parallel_config['zero']

        base_parallel_config['num_hidden_layers'] = int(base_parallel_config['num_hidden_layers'])
        base_parallel_config['ngpus'] = int(base_parallel_config['ngpus'])
        base_parallel_config['num_nodes'] = len(self.node_info.items())
        base_parallel_config['global_ngpus'] = DEVICES_PER_NODE * base_parallel_config['num_nodes']
        base_parallel_config['global_batch_size'] = int(base_parallel_config['global_batch_size'])
        #hetero
        rank_to_device_mapping = {}
        # rank_to_device_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}
        for i in range(base_parallel_config['ngpus']):
            rank_to_device_mapping[i] = i
        suspended_rank_list = []
        unused_rank_list = []
        hetero_data = True
        hetero_layers = []
        hetero_stages = []
        hetero_micro_batch_num_list = []
        for pipeline in range(base_parallel_config['dcp']):
            hetero_stages.append(base_parallel_config['pp'])
            pipeline_layers = []
            for stage in range(base_parallel_config['pp']):
                pipeline_layers.append(base_parallel_config['num_hidden_layers'] //  \
                                       base_parallel_config['pp'])
            hetero_layers.append(pipeline_layers)
            hetero_micro_batch_num_list.append(base_parallel_config["global_batch_size"] //  \
                                               base_parallel_config["dcp"])
        
        base_parallel_config["rank_to_device_mapping"] = rank_to_device_mapping
        base_parallel_config["suspended_rank_list"] = suspended_rank_list
        base_parallel_config["unused_rank_list"] = unused_rank_list
        base_parallel_config["hetero_data"] = hetero_data
        base_parallel_config["hetero_layers"] = hetero_layers
        base_parallel_config["hetero_stages"] = hetero_stages
        base_parallel_config["hetero_micro_batch_num_list"] = hetero_micro_batch_num_list

        self.base_config = base_parallel_config
        # print(f"base_config:{self.base_config}")
    
    def parse_node_info(self):
        self.parse_gpu_info()
                    
        self.trainer_ctx_config["memory_safe_gap"] = 4096.0
        self.trainer_ctx_config["straggler_threshold"] = 1.2
        self.trainer_ctx_config["straggler_safe_gap"] = 0.3
        self.trainer_ctx_config["top_k"] = 10

        print(f"trainer_ctx_config:{self.trainer_ctx_config}")

        self.trainer_ctxs = TrainerCtxs(
            bf16=True,
            hetero_tp_alpha=self.trainer_ctx_config["hetero_tp_alpha"],
            hetero_tp_weight=self.trainer_ctx_config["hetero_tp_weight"],
            normal_layers=self.base_config['num_hidden_layers'] // self.base_config['pp'],
            normal_mbn=self.base_config['global_batch_size'] // self.base_config['dcp'],
            normal_compute_time=self.trainer_ctx_config["normal_compute_time"],
            memory_k=self.trainer_ctx_config["memory_k"],
            memory_extra=self.trainer_ctx_config["memory_extra"],
            memory_embedding=self.trainer_ctx_config["memory_embedding"],
            # memory_d=memory_d,
            memory_bound=self.trainer_ctx_config["memory_bound"] * \
                         self.trainer_ctx_config["memory_percent"],
            memory_safe_gap=self.trainer_ctx_config["memory_safe_gap"],
            straggler_threshold=self.trainer_ctx_config["straggler_threshold"],
            straggler_safe_gap=self.trainer_ctx_config["straggler_safe_gap"],
            top_k=self.trainer_ctx_config["top_k"]
        )
        # hetero_layers = [[16, 16],[16, 16]]
        # hetero_stages = [2,2]
        # hetero_micro_batch_num_list = [32,32]
        self.strategy_args = TrainerStrategyArgs(
            dp = self.base_config['dcp'],
            tp = self.base_config['tp'],
            pp = self.base_config['pp'],
            zero = self.base_config['zero'],
            rank_to_device_mapping=self.base_config['rank_to_device_mapping'],
            suspended_rank_list=self.base_config['suspended_rank_list'],
            unused_rank_list=self.base_config['unused_rank_list'],
            hetero_data=self.base_config['hetero_data'],
            hetero_layers=self.base_config['hetero_layers'],
            hetero_stages=self.base_config['hetero_stages'],
            hetero_micro_batch_num_list=self.base_config['hetero_micro_batch_num_list']
        )

        suspended_devices_sr = {}
        unused_devices = []

        used_devices_sr = {}
        print(self.available_gpus)
        ptr = 0
        for k, v in self.available_gpus.items():
            node = v['gpus']
            # we suppose every node is dead at first
            for i in range(DEVICES_PER_NODE):
                used_devices_sr[ptr + i] = 10086
            for gpu in node:
                used_devices_sr[gpu['idx']] = 1
            ptr += DEVICES_PER_NODE
        # for i in range():
        #     used_devices_sr[i] = 1
        print(f"used_devices_sr:{used_devices_sr}\n" + \
              f"suspended_devices_sr:{suspended_devices_sr}\n" + \
              f"unused_devices:{unused_devices}")

        self.init_strategy_model(self.trainer_ctxs, self.strategy_args,
                                 used_devices_sr, suspended_devices_sr,
                                 unused_devices)

    def parse_gpu_info(self):
        self.trainer_ctx_config = {}

        self.trainer_ctx_config["memory_percent"] = 0.98
        self.trainer_ctx_config["available_memory_percent"] = 0

        self.trainer_ctx_config["hetero_tp_alpha"] = [1.0, 1.91, 3.77, 7.35] 

        self.trainer_ctx_config["hetero_tp_weight"] = [1.0, 1.0, 1.0, 1.0]
        self.trainer_ctx_config["normal_compute_time"] = 1

        # model related params(just an example)
        self.trainer_ctx_config["memory_k"] = [4240, 3930, 3630, 3338, 3068, 2830, 2597, 2438]  
        self.trainer_ctx_config["memory_k"] = [0, 0, 0, 0, 0, 0, 0, 0]
        self.trainer_ctx_config["memory_embedding"] = 500.0
        self.trainer_ctx_config["memory_extra"] = 4900.0

        #GPU related params
        self.trainer_ctx_config["memory_bound"] = 1000000.0
        self.available_gpus = {}
        offset = 0
        for k, v in self.node_info.items():
            gpu_info = v['gpu_info']
            self.available_gpus[k] = {}
            self.available_gpus[k]['addr'] = v['addr']
            self.available_gpus[k]['hostname'] = v['hostname']
            available_gpu = []
            for idx, item in enumerate(gpu_info):
                if (item['remain_percent'] > self.trainer_ctx_config["available_memory_percent"]):
                    if (item['total_memory'] < self.trainer_ctx_config["memory_bound"]):
                        self.trainer_ctx_config["memory_bound"] = item['total_memory']
                    device_idx = int(item['device_idx'])
                    available_gpu.append({"idx": device_idx + offset,
                                          "local_idx": device_idx,
                                          "memory": item['free_memory']})
            self.available_gpus[k]['ngpu'] = len(gpu_info)
            self.available_gpus[k]['agpu'] = len(available_gpu)
            self.available_gpus[k]['gpus'] = available_gpu
            offset += DEVICES_PER_NODE
        print("AVAILABLE_GPUS:", self.available_gpus)

    def available_gpu_info(self):
        if self.available_gpus is None:
            if self.cmd == "":
                self.parse_gpu_info()
            else:
                self.parse_node_info()
        return self.available_gpus

    def neglect_for_homo(self):
        for k in self.homo_neglect_keys:
            self.parallel_config[k] = None
    
    def strategy_example(self):
        self.parse_node_info()
        dp = 2
        tp = 2
        pp = 2
        layers = 32
        mbn = 64 # 512
        memory_percent = 0.95
        hetero_tp_alpha = [1.0, 1.91, 3.77, 7.35]
        hetero_tp_weight = [1.0, 1.0, 1.0, 1.0]
        normal_compute_time = 1 # 不重要
        # memory_k = [4240, 3930, 3630, 3338, 3068, 2830, 2597, 2438]
        memory_k = [0, 0, 0, 0, 0, 0, 0, 0]  
        memory_embedding = 500.0
        memory_extra = 4900.0
        memory_bound = 81252.0
        memory_safe_gap = 4096.0
        straggler_threshold = 1.2
        straggler_safe_gap = 0.3
        top_k = 10
        zero = True
        rank_to_device_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}
        suspended_rank_list = []
        unused_rank_list = []
        hetero_data = True
        hetero_layers = [[16, 16],[16, 16]]
        hetero_stages = [2,2]
        hetero_micro_batch_num_list = [32,32]

        suspended_devices_sr = {}
        unused_devices = []

        used_devices_sr = {}
        for i in range(8):
            used_devices_sr[i] = 1
        for i in range(8, 8):
            used_devices_sr[i] = 10086


        ctxs = TrainerCtxs(
            bf16=True,
            hetero_tp_alpha=hetero_tp_alpha,
            hetero_tp_weight=hetero_tp_weight,
            normal_layers=layers // pp,
            normal_mbn=mbn // dp,
            normal_compute_time=normal_compute_time,
            memory_k=memory_k,
            memory_extra=memory_extra,
            memory_embedding=memory_embedding,
            # memory_d=memory_d,
            memory_bound=memory_bound * memory_percent,
            memory_safe_gap=memory_safe_gap,
            straggler_threshold=straggler_threshold,
            straggler_safe_gap=straggler_safe_gap,
            top_k=top_k
        )

        strategy_args = TrainerStrategyArgs(
            dp=dp,
            tp=tp,
            pp=pp,
            zero=zero,
            rank_to_device_mapping=rank_to_device_mapping,
            suspended_rank_list=suspended_rank_list,
            unused_rank_list=unused_rank_list,
            hetero_data=hetero_data,
            hetero_layers=hetero_layers,
            hetero_stages=hetero_stages,
            hetero_micro_batch_num_list=hetero_micro_batch_num_list
        )

        self.init_strategy_model(ctxs, strategy_args, used_devices_sr,
                                 suspended_devices_sr, unused_devices)
    
    def init_strategy_model(
        self, 
        ctxs: TrainerCtxs,
        old_strategy_args: TrainerStrategyArgs, 
        used_devices_sr: Dict[int, float], 
        suspended_devices_sr: Dict[int, float],
        unused_devices: List[int]
    ):
        self.strategy_model = StrategyModel(ctxs, old_strategy_args, 
                                            used_devices_sr, 
                                            suspended_devices_sr,
                                            unused_devices)
    
    def get_gpu_num(self):
        if self.available_gpus is None:
            self.parse_node_info()
        num_gpus = 0
        for k, v in self.available_gpus.items():
            num_gpus += len(v['gpus'])
        return num_gpus

    def search_best_stategy(self):
        strategies, ds_parallel_configs = self.strategy_model.make_plans()
        # for idx, _ in enumerate(strategies):
        #     print(f"idx:{idx}, strategy:{strategies[idx]}")
        print(f"strategy:{strategies[0]}")
        self.hetero = True
        parallel_config = {}
        parallel_config['dp'] = strategies[0].dp
        parallel_config['tp'] = strategies[0].tp
        parallel_config['pp'] = strategies[0].pp
        parallel_config['cp'] = 1
        parallel_config['cp_list'] = str([1] * strategies[0].dp).replace(" ","")
        ngpu = parallel_config['dp'] * parallel_config['tp'] * \
               parallel_config['pp'] * parallel_config['cp']
        parallel_config['num_gpus'] = ngpu
        # parallel_config['global_ngpus'] = self.base_config['global_ngpus']
        parallel_config['num_layers'] = get_argument(self.train_cmd, "--num_hidden_layers")
        parallel_config['zero'] = strategies[0].zero
        recompute_layers = []

        for i in range(parallel_config['dp'] * parallel_config['cp']):
            recompute_layers.append([])
        parallel_config['recompute_layers'] = recompute_layers
        hetero_layers = ""
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result
        num_layers = 0
        hetero_layers_list = flatten(strategies[0].hetero_layers)
        for idx, item in enumerate(hetero_layers_list):
            if idx > 0:
                hetero_layers += "," + str(item)
                if (num_layers != item):
                    self.hetero = True
            else:
                hetero_layers += str(item)
                num_layers = item
        parallel_config['hetero_layers'] = hetero_layers
        hetero_stages = strategies[0].hetero_stages
        num_stage = 0
        for idx, item in enumerate(hetero_stages):
            if idx == 0:
                num_stage = item
            else:
                if item != num_stage:
                    self.hetero = True
                    break
        parallel_config['hetero_stages'] = str(hetero_stages).replace(" ","")
        unused_ranks = strategies[0].suspended_rank_list
        unused_ranks.extend(strategies[0].unused_rank_list)
        parallel_config['unused_rank'] = str(unused_ranks).replace(" ","")
        parallel_config['rank_to_device_mapping'] = str(strategies[0].rank_to_device_mapping).replace(" ","")
        self.parallel_config = parallel_config
        print("PARA_CONFIG:", self.parallel_config)
        print(f"CMD:{self.cmd}")

        strategy_dict = {}
        strategy_dict['ngpus'] = parallel_config['num_gpus']
        strategy_dict['global_ngpus'] = self.base_config['global_ngpus']
        strategy_dict['cp_list'] = str([1] * strategies[0].dp).replace(" ","")
        strategy_dict['hetero'] = True
        seq_len = None
        tmp = get_argument(self.train_cmd, "--global_seq_len")
        if tmp != False:
            seq_len = int(tmp)
        else:
            tmp = get_argument(self.train_cmd, "--seq_len_list")
            assert(tmp != False)
            seq_len = int(tmp[0])
        # strategy_dict['global_batch_size'] = (self.base_config['global_batch_size'] * \
        #                                       strategies[0].dp) // self.base_config['dcp']
        strategy_dict['seq_len_list'] = str([seq_len] * strategies[0].dp).replace(" ","")
        strategy_dict['hetero_stage_gpus'] = strategies[0].tp
        strategy_dict['hetero_stages'] = str(hetero_stages).replace(" ","")
        strategy_dict['micro_batch_num_list'] = str(strategies[0].hetero_micro_batch_num_list).replace(" ","")
        # ori_rank_to_device_mapping = strategies[0].rank_to_device_mapping
        # ori_unused_ranks_set = set(strategies[0].suspended_rank_list)
        # ordered_ranks = sorted(ori_rank_to_device_mapping.keys())
        # modifie_rank_to_device_mapping = {}
        # device_ids = []
        # for rank in ordered_ranks:
        #     if (rank not in ori_unused_ranks_set):
        #         device_ids.append(ori_rank_to_device_mapping[rank])
        # ordered_device_ids = sorted(device_ids)
        # ptr = 0
        # for rank in ordered_ranks:
        #     if (rank not in ori_unused_ranks_set):
        #         modifie_rank_to_device_mapping[rank] = ordered_device_ids[ptr]
        #         ptr += 1
        #     else:
        #         modifie_rank_to_device_mapping[rank] = ori_rank_to_device_mapping[rank]
        # print(f"ori:{ori_rank_to_device_mapping}"
        #       f"\nnew:{modifie_rank_to_device_mapping}")
        strategy_dict['rank_to_device_mapping'] = str(strategies[0].rank_to_device_mapping).replace(" ","")
        # strategy_dict['rank_to_device_mapping'] = str(modifie_rank_to_device_mapping).replace(" ","")
        strategy_dict['unused_rank'] = str(strategies[0].suspended_rank_list).replace(" ","")
        self.strategy_dict = strategy_dict
        print(f"Strategy_dict:{self.strategy_dict}")
        return parallel_config, strategy_dict
    
    def generate_parallel_config(self, log_path, ini_cmd = None):
        hetu_dir = os.path.abspath(__file__)
        for i in range(4):
            hetu_dir = os.path.dirname(hetu_dir)
        generator_path = None
        self.hetero = True
        if self.hetero:
            generator_path = hetu_dir + \
                             "/" + DS_CONFIG_PATH + "/generate_gpt_hetero_4d_config.py"
        else:
            self.neglect_for_homo()
            generator_path = hetu_dir + \
                             "/" + DS_CONFIG_PATH + "/generate_gpt_4d_config.py"
        generate_cmd = "python3 " + generator_path
        for k, v in self.parallel_config.items():
            if v is None:
                continue
            if type(v) == bool:
                if v == True:
                    generate_cmd = generate_cmd + " --" + str(k)
            else:
                if isinstance(v, (list, dict, str)):
                    v = "\"" + str(v).replace(" ","") + "\""
                generate_cmd = generate_cmd + " --" + str(k) + " " + str(v)
        # print("HETU_DIR:", hetu_dir)
        print("GENERATE_CONFIG_CMD:", generate_cmd)
        # result = subprocess.run([generate_cmd], capture_output=True, text=True, shell=True, check=True)

        agpus = self.available_gpu_info()
        print(f"agpus:{agpus}")
        for k, v in agpus.items():
            sys_pssh_client = paramiko.SSHClient()
            sys_pssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
            # print(f"NODE:{node}")
            try:
                sys_pssh_client.connect(v['addr'], port=22, timeout=TIMEOUT)
                print(f"NODE:{v['addr']} connected.")
                log_path = os.path.join(os.getcwd(), log_path)
                mkdir_cmd = "mkdir -p " + log_path
                print(mkdir_cmd)
                _, run_log, error_log = sys_pssh_client.exec_command(mkdir_cmd)
                generate_cmd = ini_cmd + " && " + generate_cmd
                _, run_log, error_log = sys_pssh_client.exec_command(generate_cmd)
                run_log = run_log.read().decode('utf-8').strip()
                error_log  = error_log.read().decode('utf-8').strip()
                # sys_pssh_client.close()
                print(f"gen_cmd:{generate_cmd}")
                # _, run_log, error_log = sys_pssh_client.exec_command("pwd")
                print(f"run_log:{run_log}\n"
                        f"error_log:{error_log}")
            except:
                print(f"NODE:{v['addr']} disconnected.")
                sys_pssh_client.close()
                continue


        tp = self.parallel_config['tp']
        pp = self.parallel_config['pp']
        if self.hetero:
            # print(self.parallel_config['cp_list'], type(self.parallel_config['cp_list']),
            #       json.loads(self.parallel_config['cp_list']))
            if isinstance(self.parallel_config['cp_list'], str):
                self.parallel_config['cp_list'] = json.loads(self.parallel_config['cp_list'])
            dcp = sum(self.parallel_config['cp_list'])
            self.strategy_dict['ds_parallel_config'] = hetu_dir + \
                                                       f"/"+ DS_CONFIG_PATH + "/hetero/" + \
                                                       f"dcp{dcp}_tp{tp}_pp{pp}.json"
        else:
            dcp = self.parallel_config['dp'] * self.parallel_config['cp']
            self.strategy_dict['ds_parallel_config'] = hetu_dir + \
                                                       f"/"+ DS_CONFIG_PATH + "/homo/" + \
                                                       f"dcp{dcp}_tp{tp}_pp{pp}.json"
        # print("RUN:", result.stdout)
        # print("CONFIG_PATH:", self.strategy_dict['ds_parallel_config'])

    
    def replace_cmd(self, ori_cmd, create_new_log=False):
        train_cmd = get_argument(ori_cmd, "--command")
        new_train_cmd = train_cmd
        # self.generate_parallel_config()
        for k, v in self.strategy_dict.items():
            if v is None:
                continue
            if type(v) == bool and v == False:
                new_train_cmd = delete_argument(new_train_cmd, "--" + k) 
                continue
            if type(v) == bool and v == True:
                new_train_cmd = add_argument(new_train_cmd, "--" + k) 
                continue
            new_train_cmd = replace_argument(new_train_cmd, "--" + k, v)
        new_cmd = replace_argument(ori_cmd, "--command", shlex.quote(new_train_cmd))
        if create_new_log:
            pass
        else:
            new_cmd = new_cmd.replace("tee", "tee -a")
        return new_cmd

    # modify some attributes
    def renew_step(self, ori_cmd, new_num_step):
        train_cmd = get_argument(ori_cmd, "--command")
        new_train_cmd = train_cmd
        new_train_cmd = replace_argument(new_train_cmd, "--steps", str(new_num_step))
        new_cmd = replace_argument(ori_cmd, "--command", shlex.quote(new_train_cmd))
        return new_cmd
    
    def set_validation(self, ori_cmd):
        train_cmd = get_argument(ori_cmd, "--command")
        new_train_cmd = train_cmd
        new_train_cmd = add_argument(new_train_cmd, "--validation")
        new_cmd = replace_argument(ori_cmd, "--command", shlex.quote(new_train_cmd))
        return new_cmd

    def set_distributed_info(self, ori_cmd, node_idx, nodes):
        train_cmd = get_argument(ori_cmd, "--command")
        new_train_cmd = train_cmd
        new_train_cmd = replace_argument(new_train_cmd, "--node_idx", str(node_idx))
        new_train_cmd = replace_argument(new_train_cmd, "--nodes", str(nodes).replace(" ",""))
        # print(f"new_train_cmd:{new_train_cmd}\n"
        #       f"quote:{shlex.quote(new_train_cmd)}")
        new_cmd = replace_argument(ori_cmd, "--command", shlex.quote(new_train_cmd))
        # new_cmd = replace_argument(ori_cmd, "--command", "\'" + new_train_cmd + "\'")
        return new_cmd
