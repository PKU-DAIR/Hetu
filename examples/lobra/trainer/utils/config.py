from hetu.peft.lora.config import LoraConfig
from typing import List, Dict
from queue import Queue
from utils.ds_parallel_config import GPUPos
from utils import convert_strategy, generate_lora_ds_parallel_config, read_ds_parallel_config
import json
import argparse

class StrategyConfig():
    dps: List[int] = []
    tps: List[int] = []
    pps: List[int] = []
    num_scheme: int = 0
    ngpus: int = 0
    ds_parallel_configs: List[Dict] = []
    gpu_pos: Dict[int, GPUPos] = {}
    
    def __init__(self, scheme_list: List[List[int]], ngpus: int, num_hidden_layers: int, strategy_id: int=0):
        self.tps, self.pps, self.dps = [], [], []
        for strategy in scheme_list:
            tp, pp, dp = strategy
            self.tps.append(tp)
            self.pps.append(pp)
            self.dps.append(dp)
        self.num_scheme = len(scheme_list)
        self.ngpus = ngpus
        layers_tp_groups, self.gpu_pos = convert_strategy(scheme_list, ngpus, num_hidden_layers)
        self.num_pipeline = len(layers_tp_groups[0])
        config_file_path = f"./ds_parallel_config/strategy_{strategy_id}.json"
        generate_lora_ds_parallel_config(ngpus, layers_tp_groups, config_file_path)
        self.ds_parallel_configs = read_ds_parallel_config(config_file_path)
    
    def get_scheme_id(self, gpu_id):
        return self.gpu_pos[gpu_id].global_dp_id

    def get_local_dp_id(self, gpu_id):
        return self.gpu_pos[gpu_id].local_dp_id

    def get_stage_id(self, gpu_id):
        return self.gpu_pos[gpu_id].stage_id

    def get_dp_degree(self, scheme_id):
        return self.dps[scheme_id]

    def get_tp_degree(self, scheme_id):
        return self.tps[scheme_id]  

    def get_pp_degree(self, scheme_id):
        return self.pps[scheme_id]  
    
    def get_num_scheme(self):
        return self.num_scheme
    
    def get_num_gpus(self):
        return self.ngpus

class TaskConfig():
    lora_config: LoraConfig = None
    global_batch_size: int = 64
    dataset_name: str = ""
    context_length: int = 0
    json_key: str = ""
    steps: int = 10
    epochs: int = 1
    
    __params_map: Dict[str, str] = {
        "global_batch_size": "global_batch_size",
        "dataset_name": "dataset_name",
        "context_length": "context_length",
        "json_key": "json_key",
        "steps": "steps",
        "epochs": "epochs"
    }
    
    def __init__(self, config):
        for key in self.__params_map:
            setattr(self, key, config[self.__params_map[key]])
        self.lora_config = LoraConfig(rank=config['rank'], lora_alpha=config['lora_alpha'], target_modules=config['target_modules'])

class TrainerConfig():
    config_path: str = ""
    variant: str = "canonical"
    train_task_num: int = 0
    task_configs: List[TaskConfig] = []
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        print(f"Reading trainer config from {config_path}...")
        trainer_config = json.load(open(config_path, 'r'))
        self.variant = trainer_config['variant']
        self.train_task_num = trainer_config['train_task_num']
        print(f"Detected {self.train_task_num} fine-tuning tasks")
        task_config_queue = Queue()
        for value in trainer_config['task']:
            if type(value) == dict:
                task_config_queue.put(value)
        while (not task_config_queue.empty()):
            task_config = task_config_queue.get()
            for target_module in task_config['target_modules']:
                if self.variant == 'fused':
                    assert target_module in ['qkv_proj', 'o_proj', 'dense_h_to_4h', 'dense_4h_to_h']
                else:
                    assert target_module in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense_h_to_4h', 'dense_4h_to_h']
            self.task_configs.append(TaskConfig(task_config))
    
    def split_and_dump_task_configs(self, split_num: int, save_dir: str=None):
        save_dir = self.config_path[:self.config_path.rfind('/') + 1]
        trainer_config = json.load(open(self.config_path, 'r'))
        task_configs = trainer_config['task']
        task_num = len(task_configs)
        assert task_num % split_num == 0
        split_task_num = task_num // split_num
        for i in range(split_num):
            split_task_configs = task_configs[i * split_task_num: (i + 1) * split_task_num]
            split_config = {
                "variant": self.variant,
                "train_task_num": split_task_num,
                "task": [task_config for task_config in split_task_configs]
            }
            split_config_path = save_dir + self.config_path[self.config_path.rfind('/') + 1: -5] + f"_{i}.json"
            with open(split_config_path, 'w') as f:
                json.dump(split_config, f, indent=4)
            print(f"Split task configs {i} saved to {split_config_path}")
    
    def get_global_batch_size_list(self):
        return [task.global_batch_size for task in self.task_configs]

if __name__ == "__main__":
    # 从args中读取配置文件路径和分割数  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/trainer_config.json")
    parser.add_argument("--split_num", type=int, default=1)
    args = parser.parse_args()
    trainer_config = TrainerConfig(args.config_path)
    trainer_config.split_and_dump_task_configs(args.split_num)
