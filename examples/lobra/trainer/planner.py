import re
import os
import time
from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed
from pyscipopt import Model, quicksum
from trainer.utils.combine_scheme_to_strategy_candidates import combine_scheme_to_strategy_candidates

class BaseStaticPlanner(ABC):
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        ngpus,
        scheme_candidates,
        use_optimized_scheme_pool=True,
        lp_threads=64
    ):
        self.num_layers = num_layers
        self.train_task_num = train_task_num
        self.global_batch_size_list = global_batch_size_list
        self.ngpus = ngpus
        self.lp_threads = lp_threads
        self.cost_model = cost_model
        self.popt = cost_model.popt
        self.cache_estimate_times = {}
        
        self.use_optimized_scheme_pool = use_optimized_scheme_pool
        self.scheme_pool = self.get_optimized_scheme_pool(scheme_candidates)
        self.scheme_pool_size = len(self.scheme_pool)
        
        self.task_seq_lens = None
        self.mbs_map = None
        self.max_batch_time_list = None
    
    def get_optimized_scheme_pool(self, scheme_candidates):
        if not self.use_optimized_scheme_pool:
            scheme_pool = [
                (
                    scheme['tp'] * scheme['pp'],
                    (scheme['tp'], scheme['pp']),
                    scheme['max_tokens']
                ) for scheme in scheme_candidates if scheme['max_tokens'] > 0
            ]
            print(f"scheme_pool = {scheme_pool}")
            return scheme_pool

        scheme_ngpus_max_tokens = {}
        scheme_max_tokens = {}
        max_tokens_list = []
        for scheme in scheme_candidates:
            tp = scheme['tp']
            pp = scheme['pp']
            ngpus = tp * pp
            if 'max_tokens' not in scheme.keys():
                assert 'mbs' in scheme.keys() and 'seq_len' in scheme.keys()
                max_tokens = scheme['mbs'] * scheme['seq_len']
            else:
                max_tokens = scheme['max_tokens']
            if max_tokens == 0:
                continue
            max_tokens_list.append(max_tokens)
            throughput = scheme['throughput_per_gpu']
            
            # update scheme_ngpus_max_tokens
            if max_tokens not in scheme_ngpus_max_tokens.keys():
                scheme_ngpus_max_tokens[max_tokens] = {}
            if ngpus not in scheme_ngpus_max_tokens[max_tokens].keys():
                scheme_ngpus_max_tokens[max_tokens][ngpus] = (throughput, (tp, pp))
            elif throughput > scheme_ngpus_max_tokens[max_tokens][ngpus][0]:
                scheme_ngpus_max_tokens[max_tokens][ngpus] = (throughput, (tp, pp))

            # update scheme_max_tokens
            if max_tokens not in scheme_max_tokens.keys():
                scheme_max_tokens[max_tokens] = (throughput, ngpus, (tp, pp))
            elif throughput > scheme_max_tokens[max_tokens][0]:
                scheme_max_tokens[max_tokens] = (throughput, ngpus, (tp, pp))
            elif throughput == scheme_max_tokens[max_tokens][0] and \
                    ngpus < scheme_max_tokens[max_tokens][1]:
                scheme_max_tokens[max_tokens] = (throughput, ngpus, (tp, pp))
            
            scheme_pool = [(scheme[1], scheme[2], max_tokens) for max_tokens, scheme in scheme_max_tokens.items()]
            # add best scheme under different ngpus
            for max_tokens in max_tokens_list:
                ngpus_of_best_scheme = scheme_max_tokens[max_tokens][1]
                for ngpus in sorted(scheme_ngpus_max_tokens[max_tokens].keys()):
                    if ngpus >= ngpus_of_best_scheme:
                        break
                    scheme_pool.append((ngpus, scheme_ngpus_max_tokens[max_tokens][ngpus][1], max_tokens))
        print(f"scheme_pool = {scheme_pool}")
        return scheme_pool

    def get_mbs_map(self, scheme_pool, task_seq_len_range):
        mbs_map = []
        for seq_len in task_seq_len_range:
            mbs_of_seq_len = [
                self.cost_model.max_tokens[scheme[1]] // seq_len
                for scheme in scheme_pool
            ]
            mbs_map.append(mbs_of_seq_len)
        return mbs_map

    def fit_time(
        self,
        X,
        aux_fragment,
        c1, c2, c3, c4, c5, c6
    ):
        mbs, seq_len = X
        return (c1 * mbs + c2 * aux_fragment) * seq_len * seq_len + \
               (c3 * mbs + c4 * aux_fragment) * seq_len + \
               (c5 * mbs + c6 * aux_fragment)
    
    def estimate_batch_time(
        self,
        mbs,
        seq_len,
        scheme,
        aux_fragment=1
    ):
        tp = scheme[1][0]
        pp = scheme[1][1]
        return self.fit_time((mbs, seq_len), aux_fragment, *self.popt[tp]) * self.num_layers / pp

    def estimate_scheme_time(
        self,
        scheme_id,
        full_micro_batch_num_list,
        rest_sample_num_list,
        aux_bool_fragment_list,
        max_batch_time_list
    ):
        scheme = self.scheme_pool[scheme_id]
        tp = scheme[1][0]
        pp = scheme[1][1]
        return quicksum(
            self.cache_estimate_times.get((
                self.mbs_map[i][scheme_id],
                seq_len,
                tp,
                pp
            ), 0) * full_micro_batch_num_list[i][scheme_id] +
            self.estimate_batch_time(
                rest_sample_num_list[i][scheme_id],
                seq_len,
                scheme,
                aux_bool_fragment_list[i][scheme_id]
            )
            for i, seq_len in enumerate(self.task_seq_lens)
        ) + (pp - 1) * max_batch_time_list[scheme_id]

    def get_estimated_batch_time(
        self,
        mbs,
        seq_len,
        tp,
        pp
    ):
        if mbs == 0:
            return 0
        return self.fit_time((mbs, seq_len), 1, *self.popt[tp]) * self.num_layers / pp

    def print_dispatch_detail(
        self,
        dp,
        tp,
        pp,
        max_tokens,
        multi_task_batch_dispatch_map,
        scheme_id
    ):
        estimate_time = 0
        max_batch_time = 0
        print(f"-----scheme - {scheme_id}: multi task seq_len map-----")
        for task in range(self.train_task_num):
            print(f"task {task}: {multi_task_batch_dispatch_map[task][scheme_id]}")
        print(f"-----scheme - {scheme_id}: multi task seq_len map-----")
        seq_len_map = {
            seq_len : np.sum([multi_task_batch_dispatch_map[task][scheme_id][seq_len] 
                              for task in range(self.train_task_num)])
            for seq_len in self.task_seq_lens
        }
        for seq_len in self.task_seq_lens:
            mbs = max_tokens // seq_len
            sample_num = (seq_len_map[seq_len] + dp - 1) // dp
            micro_batch_num = sample_num // mbs if mbs > 0 else 0
            rest_sample_num = sample_num - mbs * micro_batch_num
            full_batch_time = self.get_estimated_batch_time(mbs, seq_len, tp, pp)
            rest_batch_time = self.get_estimated_batch_time(rest_sample_num, seq_len, tp, pp)
            if micro_batch_num > 0:
                print(f"|---(mbs, seq_len, m) = ({mbs}, {seq_len}, {micro_batch_num}): {full_batch_time / 1000}s")
            if rest_sample_num > 0:
                print(f"|---(mbs, seq_len, m) = ({rest_sample_num}, {seq_len}, 1): {rest_batch_time / 1000}s")
            estimate_time += (full_batch_time * micro_batch_num + rest_batch_time)
            max_batch_time = np.max([max_batch_time, full_batch_time if micro_batch_num > 0 else rest_batch_time])
        estimate_time += (pp - 1) * max_batch_time
        print(f"scheme - {scheme_id}: max_batch_time = {max_batch_time / 1000}s, scheme_estimate_time = {estimate_time / 1000} s")
        return estimate_time
    
    @abstractmethod
    def build_planner(self, seq_distribution):
        pass
    
    @abstractmethod
    def schedule(self, multi_task_seq_distribution):
        pass

class GroupStaticPlanner(BaseStaticPlanner):
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        ngpus,
        scheme_candidates,
        use_optimized_scheme_pool=True,
        lp_threads=64
    ):
        super(GroupStaticPlanner, self).__init__(
            cost_model,
            num_layers,
            train_task_num,
            global_batch_size_list,
            ngpus,
            scheme_candidates,
            use_optimized_scheme_pool=use_optimized_scheme_pool,
            lp_threads=lp_threads
        )

    def build_planner(self, seq_distribution):
        scheme_idx = 0
        max_tokens_list = [scheme[2] for scheme in self.scheme_pool]
        sorted_max_tokens_idx = np.argsort(max_tokens_list)
        for i, seq_len in enumerate(self.task_seq_lens):
            while seq_len > self.scheme_pool[sorted_max_tokens_idx[scheme_idx]][2]:
                scheme_idx += 1
            max_tokens_of_seq_len = self.scheme_pool[sorted_max_tokens_idx[scheme_idx]][2]
            for j in range(self.scheme_pool_size):
                if self.scheme_pool[sorted_max_tokens_idx[j]][2] != max_tokens_of_seq_len:
                    self.mbs_map[i][sorted_max_tokens_idx[j]] = 0
        model = Model("group_static_planner")
        dp = [model.addVar(lb=0, ub=self.ngpus // self.scheme_pool[i][0], vtype="I", name="dp_scheme%s" % i)
              for i in range(self.scheme_pool_size)]
        n = [[model.addVar(lb=0, ub=seq_distribution[seq_len] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.scheme_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        m = [[model.addVar(lb=0, ub=seq_distribution[seq_len] // self.mbs_map[i][j] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_micro_batch_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.scheme_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        r = [[model.addVar(lb=0, ub=self.mbs_map[i][j] - 1 if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_remain_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.scheme_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        # include complete and fragment
        aux_bool_complete = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_complete(%s, scheme%s)" % (seq_len, i)) \
                             for i in range(self.scheme_pool_size)] for seq_len in self.task_seq_lens]
        aux_bool_fragment = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_fragment(%s, scheme%s)" % (seq_len, i)) \
                             for i in range(self.scheme_pool_size)] for seq_len in self.task_seq_lens]
        # max batch time only for pp > 1
        max_batch_time = [model.addVar(lb=0, ub=self.max_batch_time_list[i], vtype="C", name="max_batch_time_scheme%s" % i) if self.scheme_pool[i][1][1] > 1 else 0 \
                          for i in range(self.scheme_pool_size)]
        aux_max = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_max(scheme%s, %s)" % (i, j)) \
                   for j in range(2 * len(self.task_seq_lens))] if self.scheme_pool[i][1][1] > 1 else [] for i in range(self.scheme_pool_size)]
        # gpu num
        model.addCons(quicksum(dp[i] * self.scheme_pool[i][0] for i in range(self.scheme_pool_size)) == self.ngpus, name="eq_gpus")
        for i, seq_len in enumerate(self.task_seq_lens):
            model.addCons(quicksum(dp[j] * n[i][j] for j in range(self.scheme_pool_size)) >= seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
            model.addCons(quicksum(dp[j] * n[i][j] for j in range(self.scheme_pool_size)) <= seq_distribution[seq_len] + np.sum([self.ngpus // self.scheme_pool[k][0] - 1 for k in range(self.scheme_pool_size)]), name="le_dispatch_seq_num_%s" % seq_len)
        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.scheme_pool_size):
                model.addCons(m[i][j] * self.mbs_map[i][j] + r[i][j] == n[i][j], name="m*b_plus_r_eq_n(%s, scheme%s)" % (seq_len, j))
        		# auxiliary variable
                model.addCons(m[i][j] >= aux_bool_complete[i][j], name="m_ge_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(m[i][j] <= aux_bool_complete[i][j] * (seq_distribution[seq_len] // self.mbs_map[i][j]), name="m_le_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(m[i][j] == 0, name="m_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
                model.addCons(r[i][j] >= aux_bool_fragment[i][j], name="r_ge_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(r[i][j] <= aux_bool_fragment[i][j] * (self.mbs_map[i][j] - 1), name="r_le_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(r[i][j] == 0, name="r_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
        # max batch time
        for i in range(self.scheme_pool_size):
            if self.scheme_pool[i][1][1] == 1:
                continue
            model.addCons(quicksum(aux_max[i][j] for j in range(2 * len(self.task_seq_lens))) == 1)
            for j, seq_len in enumerate(self.task_seq_lens):
                if self.mbs_map[j][i] == 0:
                    continue
                # complete
                model.addCons(
                    max_batch_time[i] >= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.scheme_pool[i][1][0],
                         self.scheme_pool[i][1][1]
                        ), 0) * aux_bool_complete[j][i],
                    name="max_batch_time_ge_complete(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.scheme_pool[i][1][0],
                         self.scheme_pool[i][1][1]
                        ), 0) * aux_bool_complete[j][i] + 
                    self.max_batch_time_list[i] * (1 - aux_max[i][j]),
                    name="max_batch_time_le_complete_plus_bound(scheme%s, %s)" % (i, j))
                # fragment
                model.addCons(
                    max_batch_time[i] >= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.scheme_pool[i],
                        aux_bool_fragment[j][i]
                    ),
                    name="max_batch_time_ge_fragment(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.scheme_pool[i],
                        aux_bool_fragment[j][i]
                    ) +
                    self.max_batch_time_list[i] * (1 - aux_max[i][j + len(self.task_seq_lens)]),
                    name="max_batch_time_le_fragment_plus_bound(scheme%s, %s)" % (i, j)
                )

        # Set objective function
        objvar = model.addVar(name="objVar", vtype="C", lb=None, ub=None)
        model.setObjective(objvar, "minimize")
        for i in range(self.scheme_pool_size):
            model.addCons(objvar >= self.estimate_scheme_time(i, m, r, aux_bool_fragment, max_batch_time))
        return model

    def schedule(self, multi_task_seq_distribution):
        task_seq_len_range = set()
        for i in range(self.train_task_num):
            task_seq_len_range = task_seq_len_range.union(set(multi_task_seq_distribution[i].keys()))
        self.task_seq_lens = sorted(list(task_seq_len_range))
        self.mbs_map = self.get_mbs_map(self.scheme_pool, self.task_seq_lens)
        self.max_batch_time_list = [np.max([self.get_estimated_batch_time(self.mbs_map[j][i], seq_len, tp=scheme[1][0], pp=scheme[1][1]) \
                                    for j, seq_len in enumerate(self.task_seq_lens)]) for scheme in self.scheme_pool]
        self.cache_estimate_times = {}
        for scheme_id, scheme in enumerate(self.scheme_pool):
            tp = scheme[1][0]
            pp = scheme[1][1]
            for seq_len_idx, seq_len in enumerate(self.task_seq_lens):
                mbs = self.mbs_map[seq_len_idx][scheme_id]
                if mbs == 0:
                    continue
                self.cache_estimate_times[(mbs, seq_len, tp, pp)] = self.fit_time((mbs, seq_len), 1, *self.popt[tp]) * self.num_layers / pp
    	# for each task, round sample num of each seq len up
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        multi_task_seq_num_distribution = {task_id: {seq_len : 0 for seq_len in self.task_seq_lens} for task_id in range(self.train_task_num)}
        max_seq_len = max(self.task_seq_lens)
        
        for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
            for seq_len, p in task_seq_distribution.items():
                seq_num = round(p * self.global_batch_size_list[i])
                multi_task_seq_num_distribution[i][seq_len] = seq_num
                seq_distribution_across_tasks[seq_len] += seq_num
        seq_distribution_across_tasks = {s : num for s, num in seq_distribution_across_tasks.items()}
        
        # ensure there exists at least one scheme for max seq len
        if seq_distribution_across_tasks[max_seq_len] == 0:
            seq_distribution_across_tasks[max_seq_len] = 1
            for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
                if max_seq_len in task_seq_distribution.keys():
                    num = task_seq_distribution[max_seq_len] * self.global_batch_size_list[i]
                    if num > 0:
                        multi_task_seq_num_distribution[i][max_seq_len] = 1
                        break

        start_time = time.time()
        model = self.build_planner(seq_distribution_across_tasks)
        # Set model param
        model.setIntParam("lp/threads", self.lp_threads)
        model.setIntParam("parallel/maxnthreads", self.lp_threads)
        # model.setRealParam("limits/gap", 1e-3)
        # model.setRealParam('limits/time', 180)
        model.hideOutput()
        model.optimize()
        try:
            cost_time = model.getObjVal() / 1000
        except:
            end_time = time.time()
            print("No solution found")
            exit(-1)
        end_time = time.time()
        
        # get ds config, max tokens
        strategy_config = {
            'scheme_list': [],
            'max_tokens_list': [],
            'num_scheme': 0
        }
        for v in model.getVars():
            if "dp" in v.name and round(model.getVal(v)) > 0:
                dp = round(model.getVal(v))
                scheme_idx = int(re.search(r'\d+', v.name).group())
                scheme = self.scheme_pool[scheme_idx]
                tp, pp = scheme[1]
                max_tokens = scheme[2]
                strategy_config["scheme_list"].append((tp, pp, dp))
                strategy_config["max_tokens_list"].append(max_tokens)
                strategy_config["num_scheme"] += 1

        print(f"Static batch planner takes {end_time - start_time:.4f}s to get strategy config, with {strategy_config['num_scheme']} schemes as follows:")
        for i in range(strategy_config['num_scheme']):
            print(f"scheme {i}: dp = {strategy_config[i][2]}, tp = {strategy_config[i][0]}, pp = {strategy_config[i][1]}, max_tokens = {strategy_config['max_tokens_list'][i]}")
        print(f"Max scheme time cost: {cost_time:.4f}s")
        return strategy_config, end_time - start_time

class BalanceStaticPlanner(BaseStaticPlanner):
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        ngpus,
        scheme_candidates,
        use_optimized_scheme_pool=True,
        lp_threads=64
    ):
        super(BalanceStaticPlanner, self).__init__(
            cost_model,
            num_layers,
            train_task_num,
            global_batch_size_list,
            ngpus,
            scheme_candidates,
            use_optimized_scheme_pool=use_optimized_scheme_pool,
            lp_threads=lp_threads
        )

    def build_planner(self, seq_distribution):
        max_seq_len = max(seq_distribution.keys())
        align_max_seq_len = 2 ** int(np.ceil(np.log2(max_seq_len)))
        max_seq_len_scheme_num = 0
        for scheme in self.scheme_pool:
            if scheme[2] == align_max_seq_len:
                max_seq_len_scheme_num += 1
        model = Model("balanced_static_planner")
        dp = [model.addVar(
                lb=1 if max_seq_len_scheme_num == 1 and \
                        self.scheme_pool[i][2] == align_max_seq_len else 0,
                ub=self.ngpus // self.scheme_pool[i][0]
                    if self.scheme_pool[i][2] <= align_max_seq_len \
                    else 0,
                vtype="I", name="dp_scheme%s" % i) \
                for i in range(self.scheme_pool_size)
        ]
        n = [[model.addVar(lb=0, ub=seq_distribution[seq_len] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.scheme_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        m = [[model.addVar(lb=0, ub=seq_distribution[seq_len] // self.mbs_map[i][j] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_micro_batch_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.scheme_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        r = [[model.addVar(lb=0, ub=self.mbs_map[i][j] - 1 if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_remain_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.scheme_pool_size)] for i, seq_len in enumerate(self.task_seq_lens)]
        # include complete and fragment
        aux_bool_complete = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_complete(%s, scheme%s)" % (seq_len, i)) \
                             for i in range(self.scheme_pool_size)] for seq_len in self.task_seq_lens]
        aux_bool_fragment = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_fragment(%s, scheme%s)" % (seq_len, i)) \
                             for i in range(self.scheme_pool_size)] for seq_len in self.task_seq_lens]
        # max batch time only for pp > 1
        max_batch_time = [model.addVar(lb=0, ub=self.max_batch_time_list[i], vtype="C", name="max_batch_time_scheme%s" % i) if self.scheme_pool[i][1][1] > 1 else 0 \
                          for i in range(self.scheme_pool_size)]
        aux_max = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_max(scheme%s, %s)" % (i, j)) \
                   for j in range(2 * len(self.task_seq_lens))] if self.scheme_pool[i][1][1] > 1 else [] for i in range(self.scheme_pool_size)]
        # gpu num
        model.addCons(quicksum(dp[i] * self.scheme_pool[i][0] for i in range(self.scheme_pool_size)) == self.ngpus, name="eq_gpus")
        for i, seq_len in enumerate(self.task_seq_lens):
            model.addCons(quicksum(dp[j] * n[i][j] for j in range(self.scheme_pool_size)) >= seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
            model.addCons(quicksum(dp[j] * n[i][j] for j in range(self.scheme_pool_size)) <= seq_distribution[seq_len] + np.sum([self.ngpus // self.scheme_pool[k][0] - 1 for k in range(self.scheme_pool_size)]), name="le_dispatch_seq_num_%s" % seq_len)
        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.scheme_pool_size):
                model.addCons(m[i][j] * self.mbs_map[i][j] + r[i][j] == n[i][j], name="m*b_plus_r_eq_n(%s, scheme%s)" % (seq_len, j))
        		# auxiliary variable
                model.addCons(m[i][j] >= aux_bool_complete[i][j], name="m_ge_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(m[i][j] <= aux_bool_complete[i][j] * (seq_distribution[seq_len] // self.mbs_map[i][j]), name="m_le_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(m[i][j] == 0, name="m_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
                model.addCons(r[i][j] >= aux_bool_fragment[i][j], name="r_ge_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(r[i][j] <= aux_bool_fragment[i][j] * (self.mbs_map[i][j] - 1), name="r_le_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(r[i][j] == 0, name="r_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
        # max batch time
        for i in range(self.scheme_pool_size):
            if self.scheme_pool[i][1][1] == 1:
                continue
            model.addCons(quicksum(aux_max[i][j] for j in range(2 * len(self.task_seq_lens))) == 1)
            for j, seq_len in enumerate(self.task_seq_lens):
                if self.mbs_map[j][i] == 0:
                    continue
                # complete
                model.addCons(
                    max_batch_time[i] >= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.scheme_pool[i][1][0],
                         self.scheme_pool[i][1][1]
                        ), 0) * aux_bool_complete[j][i],
                    name="max_batch_time_ge_complete(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.scheme_pool[i][1][0],
                         self.scheme_pool[i][1][1]
                        ), 0) * aux_bool_complete[j][i] + 
                    self.max_batch_time_list[i] * (1 - aux_max[i][j]),
                    name="max_batch_time_le_complete_plus_bound(scheme%s, %s)" % (i, j))
                # fragment
                model.addCons(
                    max_batch_time[i] >= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.scheme_pool[i],
                        aux_bool_fragment[j][i]
                    ),
                    name="max_batch_time_ge_fragment(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.scheme_pool[i],
                        aux_bool_fragment[j][i]
                    ) +
                    self.max_batch_time_list[i] * (1 - aux_max[i][j + len(self.task_seq_lens)]),
                    name="max_batch_time_le_fragment_plus_bound(scheme%s, %s)" % (i, j)
                )

        # Set objective function
        objvar = model.addVar(name="objVar", vtype="C", lb=None, ub=None)
        model.setObjective(objvar, "minimize")
        for i in range(self.scheme_pool_size):
            model.addCons(objvar >= self.estimate_scheme_time(i, m, r, aux_bool_fragment, max_batch_time))
        return model
    
    def schedule(self, multi_task_seq_distribution):
        task_seq_len_range = set()
        for i in range(self.train_task_num):
            task_seq_len_range = task_seq_len_range.union(set(multi_task_seq_distribution[i].keys()))
        self.task_seq_lens = sorted(list(task_seq_len_range))
        self.mbs_map = self.get_mbs_map(self.scheme_pool, self.task_seq_lens)
        self.max_batch_time_list = [np.max([self.get_estimated_batch_time(self.mbs_map[j][i], seq_len, tp=scheme[1][0], pp=scheme[1][1]) \
                                    for j, seq_len in enumerate(self.task_seq_lens)]) for scheme in self.scheme_pool]
        self.cache_estimate_times = {}
        for scheme_id, scheme in enumerate(self.scheme_pool):
            tp = scheme[1][0]
            pp = scheme[1][1]
            for seq_len_idx, seq_len in enumerate(self.task_seq_lens):
                mbs = self.mbs_map[seq_len_idx][scheme_id]
                if mbs == 0:
                    continue
                self.cache_estimate_times[(mbs, seq_len, tp, pp)] = self.fit_time((mbs, seq_len), 1, *self.popt[tp]) * self.num_layers / pp
    	# for each task, round sample num of each seq len up
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        multi_task_seq_num_distribution = {task_id: {seq_len : 0 for seq_len in self.task_seq_lens} for task_id in range(self.train_task_num)}
        max_seq_len = max(self.task_seq_lens)
        
        for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
            for seq_len, p in task_seq_distribution.items():
                seq_num = round(p * self.global_batch_size_list[i])
                multi_task_seq_num_distribution[i][seq_len] = seq_num
                seq_distribution_across_tasks[seq_len] += seq_num
        seq_distribution_across_tasks = {s : num for s, num in seq_distribution_across_tasks.items()}
        
        # ensure there exists at least one scheme for max seq len
        if seq_distribution_across_tasks[max_seq_len] == 0:
            seq_distribution_across_tasks[max_seq_len] = 1
            for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
                if max_seq_len in task_seq_distribution.keys():
                    num = task_seq_distribution[max_seq_len] * self.global_batch_size_list[i]
                    if num > 0:
                        multi_task_seq_num_distribution[i][max_seq_len] = 1
                        break

        start_time = time.time()
        model = self.build_planner(seq_distribution_across_tasks)
        # Set model param
        model.setIntParam("lp/threads", self.lp_threads)
        model.setIntParam("parallel/maxnthreads", self.lp_threads)
        # model.setRealParam("limits/gap", 1e-3)
        # model.setRealParam('limits/time', 180)
        model.hideOutput()
        model.optimize()
        try:
            cost_time = model.getObjVal() / 1000
        except:
            end_time = time.time()
            print("No solution found")
            exit(-1)
        end_time = time.time()
        
        # get ds config, max tokens
        strategy_config = {
            'scheme_list': [],
            'max_tokens_list': [],
            'num_scheme': 0
        }
        for v in model.getVars():
            if "dp" in v.name and round(model.getVal(v)) > 0:
                dp = round(model.getVal(v))
                scheme_idx = int(re.search(r'\d+', v.name).group())
                scheme = self.scheme_pool[scheme_idx]
                tp, pp = scheme[1]
                max_tokens = scheme[2]
                strategy_config["scheme_list"].append((tp, pp, dp))
                strategy_config["max_tokens_list"].append(max_tokens)
                strategy_config["num_scheme"] += 1

        print(f"Static batch planner takes {end_time - start_time:.4f}s to get strategy config, with {strategy_config['num_scheme']} schemes as follows:")
        for i in range(strategy_config['num_scheme']):
            print(f"scheme {i}: dp = {strategy_config['scheme_list'][i][2]}, tp = {strategy_config['scheme_list'][i][0]}, pp = {strategy_config['scheme_list'][i][1]}, max_tokens = {strategy_config['max_tokens_list'][i]}")
        print(f"Max scheme time cost: {cost_time:.4f}s")

        return strategy_config, end_time - start_time

class PruneStaticPlanner(BaseStaticPlanner):
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        ngpus,
        scheme_candidates,
        use_optimized_scheme_pool=True,
        lp_threads=64
    ):
        super(PruneStaticPlanner, self).__init__(
            cost_model,
            num_layers,
            train_task_num,
            global_batch_size_list,
            ngpus,
            scheme_candidates,
            use_optimized_scheme_pool=use_optimized_scheme_pool,
            lp_threads=lp_threads
        )
        # auxiliary variable for dynamic batch dispatcher
        self.num_scheme = None
        self.dps = None
        self.tps = None
        self.pps = None
        self.max_tokens_list = None
        self.max_batch_time_list = None
        self.cache_estimate_times = None
        self.mbs_map = None
        self.multi_task_seq_num_distribution = None
    
    def build_planner(self, seq_distribution):
        model = Model("balance_dynamic_batch_dispatcher")
        m = [[model.addVar(lb=0, ub=seq_distribution[seq_len] // self.mbs_map[i][j] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_micro_batch_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.num_scheme)] for i, seq_len in enumerate(self.task_seq_lens)]
        n = [[model.addVar(lb=0, ub=seq_distribution[seq_len] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.num_scheme)] for i, seq_len in enumerate(self.task_seq_lens)]
        r = [[model.addVar(lb=0, ub=self.mbs_map[i][j] - 1 if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_remain_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.num_scheme)] for i, seq_len in enumerate(self.task_seq_lens)]
        # include complete and fragment
        aux_bool_complete = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_complete(%s, scheme%s)" % (seq_len, j)) \
                             for j in range(self.num_scheme)] for seq_len in self.task_seq_lens]
        aux_bool_fragment = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_fragment(%s, scheme%s)" % (seq_len, j)) \
                             for j in range(self.num_scheme)] for seq_len in self.task_seq_lens] 
        # max batch time only for pp > 1
        max_batch_time = [model.addVar(lb=0, ub=self.max_batch_time_list[i], vtype="C", name="max_batch_time(scheme%s)" % i) if self.pps[i] > 1 else 0 \
                          for i in range(self.num_scheme)]
        aux_max = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_max(scheme%s, %s)" % (i, j)) \
                   for j in range(2 * len(self.task_seq_lens))] if self.pps[i] > 1 else [] for i in range(self.num_scheme)]

        for i, seq_len in enumerate(self.task_seq_lens):
            model.addCons(quicksum(n[i][j] for j in range(self.num_scheme)) == seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
        
        # 每个策略至少分配到一条样本
        for j in range(self.num_scheme):
            model.addCons(quicksum(n[i][j] for i in range(len(self.task_seq_lens))) >= 1, name="seq_num_ge_1(scheme%s)" % j)
        
        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.num_scheme):
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] <= n[i][j] + self.dps[j] - 1, name="m*b_plus_r_eq_n(%s, scheme%s)" % (seq_len, j))
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] >= n[i][j], name="m*b_plus_r_eq_n(%s, scheme%s)" % (seq_len, j))
                # auxiliary variable
                model.addCons(m[i][j] >= aux_bool_complete[i][j], name="m_ge_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(m[i][j] <= aux_bool_complete[i][j] * (seq_distribution[seq_len] // self.mbs_map[i][j]), name="m_le_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(m[i][j] == 0, name="m_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
                model.addCons(r[i][j] >= aux_bool_fragment[i][j], name="r_ge_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(r[i][j] <= aux_bool_fragment[i][j] * (self.mbs_map[i][j] - 1), name="r_le_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(r[i][j] == 0, name="r_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
        
        # max batch time
        for i in range(self.num_scheme):
            if self.pps[i] == 1:
                continue
            model.addCons(quicksum(aux_max[i][j] for j in range(2 * len(self.task_seq_lens))) == 1)
            for j, seq_len in enumerate(self.task_seq_lens):
                if self.mbs_map[j][i] == 0:
                    continue
                # complete
                model.addCons(
                    max_batch_time[i] >= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.tps[i],
                         self.pps[i]
                        ), 0) * aux_bool_complete[j][i],
                    name="max_batch_time_ge_complete(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.tps[i],
                         self.pps[i]
                        ), 0) * aux_bool_complete[j][i] +
                        self.max_batch_time_list[i] * (1 - aux_max[i][j]),
                    name="max_batch_time_le_complete_plus_bound(scheme%s, %s)" % (i, j)
                )
                # fragment
                model.addCons(
                    max_batch_time[i] >= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.tps[i],
                        self.pps[i],
                        aux_bool_fragment[j][i]
                    ),
                    name="max_batch_time_ge_fragment(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.tps[i],
                        self.pps[i],
                        aux_bool_fragment[j][i]
                    ) + 
                    self.max_batch_time_list[i] * (1 - aux_max[i][j + len(self.task_seq_lens)]),
                    name="max_batch_time_le_fragment_plus_bound(scheme%s, %s)" % (i, j)
                )
        
        # 设置目标函数
        objvar = model.addVar(name="objVar", vtype="C", lb=None, ub=None)
        model.setObjective(objvar, "minimize")
        for j in range(self.num_scheme):
            model.addCons(objvar >= self.estimate_scheme_time(j, m, r, aux_bool_fragment, max_batch_time))
        return model
    
    def schedule(self, multi_task_seq_distribution):
        '''
        seq_distribution_map:
            global batch seq length distribution of all running tasks
        
        '''
        task_seq_len_range = set()
        for i in range(self.train_task_num):
            task_seq_len_range = task_seq_len_range.union(set(multi_task_seq_distribution[i].keys()))
        self.task_seq_lens = sorted(list(task_seq_len_range))
        # for each task, round sample num of each seq len up
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        multi_task_seq_num_distribution = {task_id: {seq_len : 0 for seq_len in self.task_seq_lens} for task_id in range(self.train_task_num)}
        max_seq_len = max(self.task_seq_lens)
        for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
            for seq_len, p in task_seq_distribution.items():
                seq_num = round(p * self.global_batch_size_list[i])
                multi_task_seq_num_distribution[i][seq_len] = seq_num
                seq_distribution_across_tasks[seq_len] += seq_num
        seq_distribution_across_tasks = {s : num for s, num in seq_distribution_across_tasks.items()}
        
        if seq_distribution_across_tasks[max_seq_len] == 0:
            seq_distribution_across_tasks[max_seq_len] = 1
            for i, task_seq_distribution in enumerate(multi_task_seq_distribution):
                if max_seq_len in task_seq_distribution.keys():
                    num = task_seq_distribution[max_seq_len] * self.global_batch_size_list[i]
                    if num > 0:
                        multi_task_seq_num_distribution[i][max_seq_len] = 1
                        break
        self.multi_task_seq_num_distribution = multi_task_seq_num_distribution
        max_seq_len = max(self.task_seq_lens)
        max_seq_len_to_2 = 2 ** int(np.ceil(np.log2(max_seq_len)))

        print("combine scheme to strategy candidates begin...")
        s_time = time.time()
        gpu_num_of_strategy_pool = np.array([strategy[0] for strategy in self.scheme_pool], dtype=np.int32)
        strategy_candidate_num, combine_dps = \
            combine_scheme_to_strategy_candidates(
                self.ngpus,
                self.scheme_pool_size,
                gpu_num_of_strategy_pool
            )
        # convert to strategy candidates and remove invalid candidates
        strategy_candidates = []
        for i in range(len(combine_dps)):
            strategy_candidates.append({})
            for j in range(len(combine_dps[i])):
                if combine_dps[i][j] not in strategy_candidates[i].keys():
                    strategy_candidates[i][combine_dps[i][j]] = combine_dps[i].count(combine_dps[i][j])
            max_tokens = 0
            for scheme_id in strategy_candidates[i].keys():
                max_tokens = max(max_tokens, self.scheme_pool[scheme_id][2])
            if max_tokens != max_seq_len_to_2:
                strategy_candidates[i] = {}
        # 删除空的strategy_candidates
        strategy_candidates = [strategy for strategy in strategy_candidates if len(strategy) > 0]
        
        def get_lower_bound_from_group_planner(strategy):
            strategy_idxs = strategy.keys()
            max_tokens_list = [self.scheme_pool[i][2] for i in strategy_idxs]
            tps_list = [self.scheme_pool[i][1][0] for i in strategy_idxs]
            pps_list = [self.scheme_pool[i][1][1] for i in strategy_idxs]
            dps_list = [strategy[i] for i in strategy_idxs]
            self.max_tokens_list = max_tokens_list
            self.dps = dps_list
            sorted_max_tokens_idxs = np.argsort(max_tokens_list)
            sorted_max_tokens_list = sorted(max_tokens_list)
            seq2strategy = {}
            for s in self.task_seq_lens:
                strategy_idx = np.searchsorted(sorted_max_tokens_list, s, side='left')
                seq2strategy[s] = sorted_max_tokens_idxs[strategy_idx]
            
            multi_task_batch_dispatch_map = {task_id : [{s : self.multi_task_seq_num_distribution[task_id][s] if seq2strategy[s] == i else 0 for s in self.task_seq_lens} for i in range(len(strategy_idxs))] for task_id in range(self.train_task_num)}
            cu_estimate_time = 0
            gpu_num = 0
            for strategy_idx in range(len(strategy_idxs)):
                estimate_time = self.get_estimate_total_time(tps_list[strategy_idx], pps_list[strategy_idx], multi_task_batch_dispatch_map, strategy_idx)
                cu_estimate_time += estimate_time * tps_list[strategy_idx] * pps_list[strategy_idx] * dps_list[strategy_idx]
                gpu_num += tps_list[strategy_idx] * pps_list[strategy_idx] * dps_list[strategy_idx]
            return cu_estimate_time / gpu_num / 1000
        
        estimate_cost = [None] * len(strategy_candidates)
        # estimate_cost = Parallel(n_jobs=min(max(1, len(strategy_combination) // 32), os.cpu_count()), prefer="processes", backend="multiprocessing")(
        #     delayed(get_lower_bound_from_group_planner)(strategy) for strategy in strategy_combination
        # )
        no_repetitive_strategy_combination_cost = []
        for strategy_idx, strategy in enumerate(strategy_candidates):
            strategy_cost = get_lower_bound_from_group_planner(strategy)
            if len(set([self.scheme_pool[i][2] for i in strategy.keys()])) == len(strategy.keys()):
                no_repetitive_strategy_combination_cost.append(strategy_cost)
            estimate_cost[strategy_idx] = strategy_cost
        
        # prune strategy candidates
        min_candidate_num = 8
        min_cost = min(no_repetitive_strategy_combination_cost)
        pruned_strategy_candidates = []
        for cost, strategy in zip(estimate_cost, strategy_candidates):
            if cost - min_cost > 0.15 * min_cost:
                continue
            pruned_strategy_candidates.append(strategy)
        if len(pruned_strategy_candidates) < min_candidate_num:
            sorted_estimate_cost_idx = np.argsort(estimate_cost)
            sorted_estimate_cost_idx = sorted_estimate_cost_idx[:min_candidate_num]
            pruned_strategy_candidates = [strategy_candidates[i] for i in sorted_estimate_cost_idx]

        cost_list = []
        def estimate_cost_after_dispatch(strategy):
            strategy_idxs = strategy.keys()
            max_tokens_list = [self.scheme_pool[i][2] for i in strategy_idxs]
            tps_list = [self.scheme_pool[i][1][0] for i in strategy_idxs]
            pps_list = [self.scheme_pool[i][1][1] for i in strategy_idxs]
            dps_list = [strategy[i] for i in strategy_idxs]
            self.max_tokens_list = max_tokens_list
            self.tps = tps_list
            self.pps = pps_list
            self.dps = dps_list
            self.num_scheme = len(strategy_idxs)
            self.mbs_map = [[max_tokens // seq_len for max_tokens in self.max_tokens_list] for seq_len in self.task_seq_lens]
            self.max_batch_time_list = [np.max([self.get_estimated_time(self.mbs_map[j][i], seq_len, self.tps[i], self.pps[i]) \
                                        for j, seq_len in enumerate(self.task_seq_lens)]) for i in range(self.num_scheme)]
            self.cache_estimate_times = {}
            for scheme_id, (tp, pp) in enumerate(zip(self.tps, self.pps)):
                for seq_len_idx, seq_len in enumerate(self.task_seq_lens):
                    mbs = self.mbs_map[seq_len_idx][scheme_id]
                    if mbs == 0:
                        continue
                    self.cache_estimate_times[(mbs, seq_len, tp, pp)] = self.fit_time((mbs, seq_len), 1, *self.popt[tp]) * self.num_layers / pp
            model = self.build_planner(seq_distribution_across_tasks)
            # Set model param
            # model.setIntParam("lp/threads", self.lp_threads)
            # model.setIntParam("parallel/maxnthreads", self.lp_threads)
            model.setLongintParam("limits/stallnodes", 5000)
            model.setRealParam("limits/gap", 5e-3)
            model.hideOutput()
            model.optimize()
            try:
                cost_time = model.getObjVal() / 1000
            except:
                print("No solution found")
                cost_list.append(1e8)
                return 1e8, None
            model_status = model.getStatus()
            dps, tps, pps, max_tokens_list = [], [], [], []
            for strategy_idx in strategy.keys():
                dp = strategy[strategy_idx]
                tp, pp = self.scheme_pool[strategy_idx][1]
                max_tokens = self.scheme_pool[strategy_idx][2]
                dps.append(dp)
                tps.append(tp)
                pps.append(pp)
                max_tokens_list.append(max_tokens)
            return cost_time, strategy, model_status
        
        results = Parallel(n_jobs=min(max(1, len(pruned_strategy_candidates) // 64), os.cpu_count()), prefer="processes")(
            delayed(estimate_cost_after_dispatch)(strategy) for strategy in pruned_strategy_candidates
        )
        e_time = time.time()
        print(f"search time = {(e_time - s_time):.3f}s")
        
        cost_idxs = np.argsort([result[0] for result in results])
        for cost_idx in cost_idxs:
            multi_strategy = results[cost_idx][1]
            if len(multi_strategy.keys()) > 1:
                continue
            _, lower_bound_of_best_strategy = get_lower_bound_from_group_planner(i, multi_strategy)
            print(f"====================")
            print(f"cost - single: {results[cost_idx][0]}, lower_bound = {lower_bound_of_best_strategy}, status = {results[cost_idx][2]}")
            for scheme_idx in multi_strategy.keys():
                scheme = self.scheme_pool[scheme_idx]
                dp = multi_strategy[scheme_idx]
                tp, pp = scheme[1]
                max_tokens = scheme[2]
                print(f"scheme - {scheme_idx}: dp = {dp}, tp = {tp}, pp = {pp}, max_tokens = {max_tokens}")
            print(f"====================")
        
        '''
        k = min(5, len(results))
        for i in range(k):
            min_cost_idx = cost_idxs[i]
            multi_strategy = results[min_cost_idx][1]
            if multi_strategy is None:
                continue
            _, lower_bound_of_best_strategy = self.get_lower_bound_from_group_planner(i, multi_strategy)
            print(f"====================")
            # print(f"cost - {i}: {cost_list[cost_idxs[i]]}")
            print(f"cost - {i}: {results[min_cost_idx][0]}, lower_bound = {lower_bound_of_best_strategy}, status = {results[min_cost_idx][2]}")
            for strategy_idx in multi_strategy.keys():
                strategy = self.strategy_pool[strategy_idx]
                dp = multi_strategy[strategy_idx]
                tp, pp = strategy[1]
                max_tokens = strategy[2]
                print(f"strategy - {strategy_idx}: dp = {dp}, tp = {tp}, pp = {pp}, max_tokens = {max_tokens}")
            print(f"====================")
        '''

        min_cost_idx = cost_idxs[0]
        multi_strategy = results[min_cost_idx][1]
        if os.environ.get("BUCKET_PLAN") == "PROFILE" and \
           os.environ.get("EXPR_EFFECTIVENESS") == "ON":
            with open('effectiveness_static.txt', 'a') as f:
                for scheme_idx in multi_strategy.keys():
                    scheme = self.scheme_pool[scheme_idx]
                    dp = multi_strategy[scheme_idx]
                    tp, pp = scheme[1]
                    max_tokens = scheme[2]
                    f.write(f"dp = {dp}, tp = {tp}, pp = {pp}, max_tokens = {max_tokens}\n")
                f.write(f"{results[min_cost_idx][0]}\n")
                f.write(f"{e_time - s_time:.3f}\n")
                f.write("\n")
        return multi_strategy, e_time - s_time

class BaseDynamicDispatcher(ABC):
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        strategy_config,
        max_tokens_list,
        scheme_id,
        local_device_idx=0,
        lp_threads=64,
    ):
        self.num_layers = num_layers
        self.train_task_num = train_task_num
        self.global_batch_size_list = global_batch_size_list
        self.max_tokens_list = max_tokens_list
        self.num_scheme = strategy_config.get_num_scheme()
        self.dps = strategy_config.dps
        self.tps = strategy_config.tps
        self.pps = strategy_config.pps
        self.lp_threads = lp_threads
        self.scheme_id = scheme_id
        self.popt = cost_model.popt
        
        self.task_seq_lens = None
        self.mbs_map = None
        self.max_batch_time_list = None
        self.running_task_ids = None
        self.cache_estimate_times = {}

        # tokens
        self.token_num = 0
        self.valid_token_num = 0
        
        # experiment only
        self.local_device_idx = local_device_idx
    
    def fit_time(
        self,
        X,
        aux_fragment,
        c1, c2, c3, c4, c5, c6
    ):
        mbs, seq_len = X
        return (c1 * mbs + c2 * aux_fragment) * seq_len * seq_len + \
               (c3 * mbs + c4 * aux_fragment) * seq_len + \
               (c5 * mbs + c6 * aux_fragment)
    
    def estimate_batch_time(
        self,
        mbs,
        seq_len,
        tp,
        pp,
        aux_fragment=1
    ):
        return self.fit_time((mbs, seq_len), aux_fragment, *self.popt[tp]) * self.num_layers / pp

    def estimate_scheme_time(
        self,
        scheme_id,
        full_micro_batch_num_list,
        rest_sample_num_list,
        aux_bool_fragment_list,
        max_batch_time_list
    ):
        return quicksum(
            self.cache_estimate_times.get((
                self.mbs_map[i][scheme_id],
                seq_len,
                self.tps[scheme_id],
                self.pps[scheme_id]
            ), 0) * full_micro_batch_num_list[i][scheme_id] +
            self.estimate_batch_time(
                rest_sample_num_list[i][scheme_id],
                seq_len,
                self.tps[scheme_id],
                self.pps[scheme_id],
                aux_bool_fragment_list[i][scheme_id]
            )
            for i, seq_len in enumerate(self.task_seq_lens)
        ) + (self.pps[scheme_id] - 1) * max_batch_time_list[scheme_id]
    
    def get_estimated_batch_time(
        self,
        mbs,
        seq_len,
        tp,
        pp
    ):
        if mbs == 0:
            return 0
        return self.fit_time((mbs, seq_len), 1, *self.popt[tp]) * self.num_layers / pp
    
    def get_estimated_scheme_time(
        self,
        tp,
        pp,
        multi_task_batch_dispatch_map,
        scheme_id
    ):
        estimate_time = 0
        max_batch_time = 0
        max_tokens = self.max_tokens_list[scheme_id]
        seq_len_map = {
            seq_len : np.sum([multi_task_batch_dispatch_map[task][scheme_id][seq_len] 
                              for task in self.running_task_ids])
            for seq_len in self.task_seq_lens
        }
        for seq_len in self.task_seq_lens:
            mbs = max_tokens // seq_len
            sample_num = (seq_len_map[seq_len] + self.dps[scheme_id] - 1) // self.dps[scheme_id]
            micro_batch_num = sample_num // mbs if mbs > 0 else 0
            rest_sample_num = sample_num - mbs * micro_batch_num
            full_batch_time = self.get_estimated_batch_time(mbs, seq_len, tp, pp)
            rest_batch_time = self.get_estimated_batch_time(rest_sample_num, seq_len, tp, pp)
            estimate_time += (full_batch_time * micro_batch_num + rest_batch_time)
            max_batch_time = np.max([max_batch_time, full_batch_time if micro_batch_num > 0 else rest_batch_time])
        estimate_time += (pp - 1) * max_batch_time
        return estimate_time

    def print_dispatch_detail(
        self,
        tp,
        pp,
        multi_task_batch_dispatch_map,
        scheme_id
    ):
        estimate_time = 0
        max_batch_time = 0
        max_tokens = self.max_tokens_list[scheme_id]
        print(f"-----scheme - {scheme_id}: multi task seq_len map-----")
        for task in self.running_task_ids:
            print(f"task {task}: {multi_task_batch_dispatch_map[task][scheme_id]}")
        print(f"-----scheme - {scheme_id}: multi task seq_len map-----")
        seq_len_map = {
            seq_len : np.sum([multi_task_batch_dispatch_map[task][scheme_id][seq_len] 
                              for task in self.running_task_ids])
            for seq_len in self.task_seq_lens
        }
        
        if os.environ.get('EXPR_CASE_STUDY') == 'ON':
            data_dispatch_pattern = os.environ.get('EXPR_DATA_DISPATCH')
            if not os.path.exists(f"case_study/{data_dispatch_pattern}-{os.environ.get('DP_BUCKET')}"):
                os.makedirs(f"case_study/{data_dispatch_pattern}-{os.environ.get('DP_BUCKET')}")
            seq_to_num = {}
            seq_to_time = {}
            seq_num = 0
            def get_bucket(seq_len):
                buckets = [2048, 4096, 8192, 16384]
                for bucket in buckets:
                    if seq_len <= bucket:
                        return bucket
        
        for seq_len in self.task_seq_lens:
            mbs = max_tokens // seq_len
            sample_num = (seq_len_map[seq_len] + self.dps[scheme_id] - 1) // self.dps[scheme_id]
            micro_batch_num = sample_num // mbs if mbs > 0 else 0
            rest_sample_num = sample_num - mbs * micro_batch_num
            full_batch_time = self.get_estimated_batch_time(mbs, seq_len, tp, pp)
            rest_batch_time = self.get_estimated_batch_time(rest_sample_num, seq_len, tp, pp)
            if micro_batch_num > 0:
                print(f"|---(mbs, seq_len, m) = ({mbs}, {seq_len}, {micro_batch_num}): {full_batch_time / 1000}s")
            if rest_sample_num > 0:
                print(f"|---(mbs, seq_len, m) = ({rest_sample_num}, {seq_len}, 1): {rest_batch_time / 1000}s")
            estimate_time += (full_batch_time * micro_batch_num + rest_batch_time)
            max_batch_time = np.max([max_batch_time, full_batch_time if micro_batch_num > 0 else rest_batch_time])
            
            if os.environ.get('EXPR_CASE_STUDY') == 'ON':
                bucket = get_bucket(seq_len)
                seq_num += sample_num
                seq_to_num[bucket] = seq_to_num.get(bucket, 0) + sample_num
                seq_to_time[bucket] = seq_to_time.get(bucket, 0) + (full_batch_time * micro_batch_num + rest_batch_time)
        estimate_time += (pp - 1) * max_batch_time
        print(f"scheme - {scheme_id}: max_batch_time = {max_batch_time / 1000}s, scheme_estimate_time = {estimate_time / 1000} s")
        
        if os.environ.get('EXPR_CASE_STUDY') == 'ON':
            seq_to_percent = {k: v / seq_num for k, v in seq_to_num.items()}
            seq_to_time = {k: v / estimate_time for k, v in seq_to_time.items()}
            local_host_name = os.environ['HETU_LOCAL_HOSTNAME']
            data_dispatch_pattern = os.environ.get('EXPR_DATA_DISPATCH')
            if scheme_id == self.scheme_id:
                with open(f"case_study/{data_dispatch_pattern}-{os.environ.get('DP_BUCKET')}/{local_host_name}-{self.local_device_idx}.txt", 'a') as f:
                    f.write(f"{seq_to_percent}\n")
                    f.write(f"{seq_to_time}\n")
        return estimate_time
    
    @abstractmethod
    def build_planner(self, seq_distribution):
        pass
    
    def schedule(self, seq_distribution_map, is_profile=False):
        '''
        seq_distribution_map:
            global batch seq length distribution of all running tasks
        
        '''
        self.running_task_ids = sorted(list(seq_distribution_map.keys()))
        for task_id in self.running_task_ids:
            for seq_len in sorted(seq_distribution_map[task_id].keys()):
                seq_distribution_map[task_id][seq_len] = int(seq_distribution_map[task_id][seq_len] * self.global_batch_size_list[task_id])
        self.task_seq_lens = sorted(seq_distribution_map[0].keys())
        self.mbs_map = [[max_tokens // seq_len for max_tokens in self.max_tokens_list] for seq_len in self.task_seq_lens]
        self.max_batch_time_list = [np.max([self.get_estimated_batch_time(self.mbs_map[j][i], seq_len, tp, pp) \
                                            for j, seq_len in enumerate(self.task_seq_lens)]) for i, (tp, pp) in enumerate(zip(self.tps, self.pps))]
        self.cache_estimate_times = {}
        for scheme_id, (tp, pp) in enumerate(zip(self.tps, self.pps)):
            for seq_len_idx, seq_len in enumerate(self.task_seq_lens):
                mbs = self.mbs_map[seq_len_idx][scheme_id]
                if mbs == 0:
                    continue
                self.cache_estimate_times[(mbs, seq_len, tp, pp)] = self.fit_time((mbs, seq_len), 1, *self.popt[tp]) * self.num_layers / pp
        # merge seq distribution of all tasks
        seq_distribution_across_tasks = {s : 0 for s in self.task_seq_lens}
        for task_id in self.running_task_ids:
            for seq_len in self.task_seq_lens:
                seq_distribution_across_tasks[seq_len] += seq_distribution_map[task_id][seq_len]
        
        start_time = time.time()
        model = self.build_planner(seq_distribution_across_tasks)
        # Set model param
        model.setIntParam("lp/threads", self.lp_threads)
        model.setIntParam("parallel/maxnthreads", self.lp_threads)
        # model.setLongintParam("limits/nodes", 100000)
        model.setLongintParam("limits/stallnodes", 6000)
        # model.setRealParam("limits/gap", 1e-3)
        model.hideOutput()
        model.optimize()
        try:
            model.getObjVal()
        except:
            print("No solution found")
            end_time = time.time()
            return {}, end_time - start_time
        end_time = time.time()
        print(f"Dynamic batch dispatcher takes {end_time - start_time:.4f}s")
        
        # get the dispatch result for all schemes
        seq_len_num_map_list = []
        for scheme_id in range(self.num_scheme):
            seq_len_num_map = {s : 0 for s in self.task_seq_lens}
            for v in model.getVars():
                for seq_len in self.task_seq_lens:
                    n_name = "seq_num(%s, scheme%s)" % (seq_len, scheme_id)
                    if n_name in v.name:
                        n_val = round(model.getVal(v))
                        if n_val < 1:
                            continue
                        seq_len_num_map[seq_len] = n_val
            seq_len_num_map_list.append(seq_len_num_map)
        
        # dispatch task-specific samples
        multi_task_batch_dispatch_map = {
            task_id : [
                {s : 0 for s in self.task_seq_lens}
                    for _ in range(self.num_scheme)
                ]
            for task_id in self.running_task_ids
        }
        for seq_len in self.task_seq_lens:
            for task_id in self.running_task_ids:
                for scheme_id in range(self.num_scheme):
                    dispatch_num = min(seq_len_num_map_list[scheme_id].get(seq_len, 0), seq_distribution_map[task_id].get(seq_len, 0))
                    if dispatch_num == 0:
                        continue
                    multi_task_batch_dispatch_map[task_id][scheme_id][seq_len] += int(dispatch_num)
                    seq_len_num_map_list[scheme_id][seq_len] -= dispatch_num
                    seq_distribution_map[task_id][seq_len] -= dispatch_num
                assert seq_distribution_map[task_id].get(seq_len, 0) == 0, \
                    f"data dispatch error: task {task_id} seq_len {seq_len} = {seq_distribution_map[task_id][seq_len]}"

        # dispatch detail
        for scheme_id in range(self.num_scheme):
            self.print_dispatch_detail(self.tps[scheme_id], self.pps[scheme_id], multi_task_batch_dispatch_map, scheme_id)
        
        # profile
        cost_time = 0
        if is_profile:
            for scheme_id in range(self.num_scheme):
                cost_time = max(
                    cost_time,
                    self.get_estimated_scheme_time(
                        self.tps[scheme_id],
                        self.pps[scheme_id],
                        multi_task_batch_dispatch_map,
                        scheme_id)
                )

        return multi_task_batch_dispatch_map, end_time - start_time, cost_time / 1000
        
class GroupDynamicDispatcher(BaseDynamicDispatcher):
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        strategy_config,
        max_tokens_list,
        scheme_id,
        local_device_idx=0,
        lp_threads=64,
    ):
        super(GroupDynamicDispatcher, self).__init__(
            cost_model,
            num_layers,
            train_task_num,
            global_batch_size_list,
            strategy_config,
            max_tokens_list,
            scheme_id,
            local_device_idx,
            lp_threads
        )
    
    def build_planner(self, seq_distribution):
        scheme_id = 0
        sorted_max_tokens_idx = np.argsort(self.max_tokens_list)
        for i, seq_len in enumerate(self.task_seq_lens):
            while seq_len > self.max_tokens_list[sorted_max_tokens_idx[scheme_id]]:
                scheme_id += 1
            max_tokens_of_seq_len = self.max_tokens_list[sorted_max_tokens_idx[scheme_id]]
            for j in range(self.num_scheme):
                if self.max_tokens_list[sorted_max_tokens_idx[j]] != max_tokens_of_seq_len:
                    self.mbs_map[i][sorted_max_tokens_idx[j]] = 0

        model = Model("group_dynamic_batch_dispatcher")
        m = [[model.addVar(lb=0, ub=seq_distribution[seq_len] // self.mbs_map[i][j] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_micro_batch_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.num_scheme)] for i, seq_len in enumerate(self.task_seq_lens)]
        n = [[model.addVar(lb=0, ub=seq_distribution[seq_len] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.num_scheme)] for i, seq_len in enumerate(self.task_seq_lens)]
        r = [[model.addVar(lb=0, ub=self.mbs_map[i][j] - 1 if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_remain_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.num_scheme)] for i, seq_len in enumerate(self.task_seq_lens)]
        # include complete and fragment
        aux_bool_complete = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_complete(%s, scheme%s)" % (seq_len, i)) \
                             for i in range(self.num_scheme)] for seq_len in self.task_seq_lens]
        aux_bool_fragment = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_fragment(%s, scheme%s)" % (seq_len, i)) \
                             for i in range(self.num_scheme)] for seq_len in self.task_seq_lens]
        # max batch time only for pp > 1
        max_batch_time = [model.addVar(lb=0, ub=self.max_batch_time_list[i], vtype="C", name="max_batch_time(scheme%s)" % i) if self.pps[i] > 1 else 0 \
                          for i in range(self.num_scheme)]
        aux_max = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_max(scheme%s, %s)" % (i, j)) \
                   for j in range(2 * len(self.task_seq_lens))] if self.pps[i] > 1 else [] for i in range(self.num_scheme)]

        for i, seq_len in enumerate(self.task_seq_lens):
            model.addCons(quicksum(n[i][j] for j in range(self.num_scheme)) == seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
        
        # 每个策略至少分配到一条样本
        for j in range(self.num_scheme):
            model.addCons(quicksum(n[i][j] for i in range(len(self.task_seq_lens))) >= 1, name="seq_num_ge_1(scheme%s)" % j)

        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.num_scheme):
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] <= n[i][j] + self.dps[j] - 1, name="m*b_plus_r_eq_n(%s, scheme%s)" % (seq_len, j))
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] >= n[i][j], name="m*b_plus_r_eq_n(%s, scheme%s)" % (seq_len, j))
                # auxiliary variable
                model.addCons(m[i][j] >= aux_bool_complete[i][j], name="m_ge_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(m[i][j] <= aux_bool_complete[i][j] * (seq_distribution[seq_len] // self.mbs_map[i][j]), name="m_le_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(m[i][j] == 0, name="m_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
                model.addCons(r[i][j] >= aux_bool_fragment[i][j], name="r_ge_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(r[i][j] <= aux_bool_fragment[i][j] * (self.mbs_map[i][j] - 1), name="r_le_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(r[i][j] == 0, name="r_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
        
        # max batch time
        for i in range(self.num_scheme):
            if self.pps[i] == 1:
                continue
            model.addCons(quicksum(aux_max[i][j] for j in range(2 * len(self.task_seq_lens))) == 1)
            for j, seq_len in enumerate(self.task_seq_lens):
                if self.mbs_map[j][i] == 0:
                    continue
                # complete
                model.addCons(
                    max_batch_time[i] >= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.tps[i],
                         self.pps[i]
                        ), 0) * aux_bool_complete[j][i],
                    name="max_batch_time_ge_complete(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.tps[i],
                         self.pps[i]
                        ), 0) * aux_bool_complete[j][i] + 
                    self.max_batch_time_list[i] * (1 - aux_max[i][j]),
                    name="max_batch_time_le_complete_plus_bound(scheme%s, %s)" % (i, j)
                )
                # fragment
                model.addCons(
                    max_batch_time[i] >= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.tps[i],
                        self.pps[i],
                        aux_bool_fragment[j][i]
                    ),
                    name="max_batch_time_ge_fragment(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.tps[i],
                        self.pps[i],
                        aux_bool_fragment[j][i]
                    ) +
                    self.max_batch_time_list[i] * (1 - aux_max[i][j + len(self.task_seq_lens)]),
                    name="max_batch_time_le_fragment_plus_bound(scheme%s, %s)" % (i, j)
                )
        
        # 设置目标函数
        objvar = model.addVar(name="objVar", vtype="C", lb=None, ub=None)
        model.setObjective(objvar, "minimize")
        for j in range(self.num_scheme):
            model.addCons(objvar >= self.estimate_scheme_time(j, m, r, aux_bool_fragment, max_batch_time))
        return model

class BalanceDynamicDispatcher(BaseDynamicDispatcher):
    def __init__(
        self,
        cost_model,
        num_layers,
        train_task_num,
        global_batch_size_list,
        strategy_config,
        max_tokens_list,
        scheme_id,
        local_device_idx=0,
        lp_threads=64,
    ):
        super(BalanceDynamicDispatcher, self).__init__(
            cost_model,
            num_layers,
            train_task_num,
            global_batch_size_list,
            strategy_config,
            max_tokens_list,
            scheme_id,
            local_device_idx,
            lp_threads
        )
    
    def build_planner(self, seq_distribution):
        model = Model("balance_dynamic_batch_dispatcher")
        m = [[model.addVar(lb=0, ub=seq_distribution[seq_len] // self.mbs_map[i][j] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_micro_batch_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.num_scheme)] for i, seq_len in enumerate(self.task_seq_lens)]
        n = [[model.addVar(lb=0, ub=seq_distribution[seq_len] if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.num_scheme)] for i, seq_len in enumerate(self.task_seq_lens)]
        r = [[model.addVar(lb=0, ub=self.mbs_map[i][j] - 1 if self.mbs_map[i][j] > 0 else 0, vtype="I", name="seq_remain_num(%s, scheme%s)" % (seq_len, j)) \
             for j in range(self.num_scheme)] for i, seq_len in enumerate(self.task_seq_lens)]
        # include complete and fragment
        aux_bool_complete = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_complete(%s, scheme%s)" % (seq_len, j)) \
                             for j in range(self.num_scheme)] for seq_len in self.task_seq_lens]
        aux_bool_fragment = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_bool_fragment(%s, scheme%s)" % (seq_len, j)) \
                             for j in range(self.num_scheme)] for seq_len in self.task_seq_lens] 
        # max batch time only for pp > 1
        max_batch_time = [model.addVar(lb=0, ub=self.max_batch_time_list[i], vtype="C", name="max_batch_time(scheme%s)" % i) if self.pps[i] > 1 else 0 \
                          for i in range(self.num_scheme)]
        aux_max = [[model.addVar(lb=0, ub=1, vtype="B", name="aux_max(scheme%s, %s)" % (i, j)) \
                   for j in range(2 * len(self.task_seq_lens))] if self.pps[i] > 1 else [] for i in range(self.num_scheme)]

        for i, seq_len in enumerate(self.task_seq_lens):
            model.addCons(quicksum(n[i][j] for j in range(self.num_scheme)) == seq_distribution[seq_len], name="ge_dispatch_seq_num_%s" % seq_len)
        
        # 每个策略至少分配到一条样本
        for j in range(self.num_scheme):
            model.addCons(quicksum(n[i][j] for i in range(len(self.task_seq_lens))) >= 1, name="seq_num_ge_1(scheme%s)" % j)
        
        # micro batch num
        for i, seq_len in enumerate(self.task_seq_lens):
            for j in range(self.num_scheme):
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] <= n[i][j] + self.dps[j] - 1, name="m*b_plus_r_eq_n(%s, scheme%s)" % (seq_len, j))
                model.addCons((m[i][j] * self.mbs_map[i][j] + r[i][j]) * self.dps[j] >= n[i][j], name="m*b_plus_r_eq_n(%s, scheme%s)" % (seq_len, j))
                # auxiliary variable
                model.addCons(m[i][j] >= aux_bool_complete[i][j], name="m_ge_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(m[i][j] <= aux_bool_complete[i][j] * (seq_distribution[seq_len] // self.mbs_map[i][j]), name="m_le_aux_bool_complete(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(m[i][j] == 0, name="m_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
                model.addCons(r[i][j] >= aux_bool_fragment[i][j], name="r_ge_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                if self.mbs_map[i][j] > 0:
                    model.addCons(r[i][j] <= aux_bool_fragment[i][j] * (self.mbs_map[i][j] - 1), name="r_le_aux_bool_fragment(%s, scheme%s)" % (seq_len, j))
                else:
                    model.addCons(r[i][j] == 0, name="r_eq_0_if_mbs_eq_0(%s, scheme%s)" % (seq_len, j))
        
        # max batch time
        for i in range(self.num_scheme):
            if self.pps[i] == 1:
                continue
            model.addCons(quicksum(aux_max[i][j] for j in range(2 * len(self.task_seq_lens))) == 1)
            for j, seq_len in enumerate(self.task_seq_lens):
                if self.mbs_map[j][i] == 0:
                    continue
                # complete
                model.addCons(
                    max_batch_time[i] >= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.tps[i],
                         self.pps[i]
                        ), 0) * aux_bool_complete[j][i],
                    name="max_batch_time_ge_complete(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.cache_estimate_times.get(
                        (self.mbs_map[j][i],
                         seq_len,
                         self.tps[i],
                         self.pps[i]
                        ), 0) * aux_bool_complete[j][i] +
                        self.max_batch_time_list[i] * (1 - aux_max[i][j]),
                    name="max_batch_time_le_complete_plus_bound(scheme%s, %s)" % (i, j)
                )
                # fragment
                model.addCons(
                    max_batch_time[i] >= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.tps[i],
                        self.pps[i],
                        aux_bool_fragment[j][i]
                    ),
                    name="max_batch_time_ge_fragment(scheme%s, %s)" % (i, j)
                )
                model.addCons(
                    max_batch_time[i] <= self.estimate_batch_time(
                        r[j][i],
                        seq_len,
                        self.tps[i],
                        self.pps[i],
                        aux_bool_fragment[j][i]
                    ) + 
                    self.max_batch_time_list[i] * (1 - aux_max[i][j + len(self.task_seq_lens)]),
                    name="max_batch_time_le_fragment_plus_bound(scheme%s, %s)" % (i, j)
                )
        
        # Set objective function
        objvar = model.addVar(name="objVar", vtype="C", lb=None, ub=None)
        model.setObjective(objvar, "minimize")
        for j in range(self.num_scheme):
            model.addCons(objvar >= self.estimate_scheme_time(j, m, r, aux_bool_fragment, max_batch_time))
        return model
