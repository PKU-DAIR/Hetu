import os
import csv
from scipy.optimize import curve_fit
from utils.logger import read_from_csv

class CostModel:
    '''
    Profile-based cost model for multi-task lora fine-tuning.
    '''
    def __init__(
        self,
        model_config,
        profile_path,
        profile_memory_path=None,
        sequence_parallel=True,
    ):
        self.popt = {}
        self.seq_len_range = None
        self.max_tokens = None
        self.tps = [1, 2, 4, 8, 16]
        self.profile_mbs = [1, 4]
        self.profile_n_layer = 2
        self.sequence_parallel = sequence_parallel
        self.model_config = model_config
        self.profile_memory_path = profile_memory_path
        self.profile_path = profile_path
        print(f"Building cost model...")
        
        # memory profile
        if os.path.exists(profile_memory_path):
            self.read_from_mem_profile(profile_memory_path)
        else:
            print(f"[WARN] Memory profile file not found, it will be used in static planner.")

        # time profile
        if os.path.exists(profile_path):
            self.read_from_time_profile(profile_path)
        else:
            print(f"[ERROR] Cannot find profile_path: {profile_path}")
            exit(-1)
    
    def read_from_mem_profile(self, profile_memory_path):
        print(f"Read profiled max tokens from {profile_memory_path}")
        max_tokens = {}
        rows = read_from_csv(profile_memory_path)
        for row in rows:
            tp = row['tp']
            pp = row['pp']
            max_tokens[(tp, pp)] = row['max_tokens']
        self.max_tokens = max_tokens

    def read_from_time_profile(self, profile_path):
        print(f"Read time cost from {profile_path}")
        mbs_dict = {}
        seq_len_dict = {}
        estimate_time_dict = {}
        self.seq_len_range = set()
        
        for tp in self.tps:
            mbs_dict[tp] = []
            seq_len_dict[tp] = []
            estimate_time_dict[tp] = []

        # 读取单层 Transformer Layer 的时间
        with open(profile_path, 'r') as f:
            reader = csv.reader(f)
            _ = next(reader)
            
            for row in reader:
                tp = int(row[0])
                seq_len = int(row[1])
                mbs = int(row[2])
                block_time = float(row[-1])
                mbs_dict[tp].append(mbs)
                seq_len_dict[tp].append(seq_len)
                estimate_time_dict[tp].append(block_time)
                self.seq_len_range.add(seq_len)
        
        for tp in self.tps:
            if len(mbs_dict[tp]) == 0:
                self.popt[tp] = None
            else:
                self.popt[tp], _ = curve_fit(self.curve_fit_func, (mbs_dict[tp], seq_len_dict[tp]), estimate_time_dict[tp])
    
    def curve_fit_func(self, X, c1, c2, c3, c4, c5, c6):
        mbs, seq_len = X
        return (c1 * mbs + c2) * seq_len * seq_len + (c3 * mbs + c4) * seq_len + (c5 * mbs + c6)

    def estimate_time(self, mbs, seq_len, tp, pp, num_micro_batches, num_layers):
        if mbs == 0 or num_micro_batches == 0:
            return 0
        return self.curve_fit_func((mbs, seq_len), *self.popt[tp]) * num_layers * (num_micro_batches + pp - 1) / pp
    
    def get_strategy_candidates(self, num_layers, num_micro_batches=16):
        strategy_candidates = []
        for (tp, pp), max_tokens in self.max_tokens.items():
            num_gpus = tp * pp
            latency = self.estimate_time(1, max_tokens, tp, pp, num_micro_batches, num_layers) / 1000
            strategy_candidate = {
                'tp': tp,
                'pp': pp,
                'max_tokens': max_tokens,
                'latency': latency,
                'throughput_per_gpu': (max_tokens * num_micro_batches) / (latency * num_gpus)
            }
            strategy_candidates.append(strategy_candidate)
        return strategy_candidates
