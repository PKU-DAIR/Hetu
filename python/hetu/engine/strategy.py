import math
import heapq
import pulp
import time
from typing import List, Dict, Any
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum
from .utils import Args, TrainerCtxs, TrainerStrategyArgs
from hetu.utils.parallel.read_ds import config_spread_zero
from hetu.models.gpt.generate_gpt_hetero_4d_config import generate_gpt_hetero_4d_config

DEVICES_PER_NODE = 8
INF_SR = 10000.0

class LayersProp(Args):
    def __init__(
        self,
        begin_layer_idx: int,
        end_layer_idx: int,
        slice_num: int,
        slices_sum: int
    ):
        self.begin_layer_idx = begin_layer_idx
        self.end_layer_idx = end_layer_idx
        self.slice_num = slice_num
        self.slices_sum = slices_sum
        
    @staticmethod    
    def calculate_iou(range1, range2):
        """
        计算两个范围的IoU
        参数:
        range1 -- 第一个范围，一个元组 (start1, end1)
        range2 -- 第二个范围，一个元组 (start2, end2)
        返回:
        IoU -- 交并比
        """
        start1, end1 = range1
        start2, end2 = range2
        # 计算交集的起始和结束点
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        # 计算交集的长度
        intersection_length = max(0, intersection_end - intersection_start)
        # 计算两个范围的长度
        range1_length = end1 - start1
        range2_length = end2 - start2
        # 计算并集的长度
        union_length = range1_length + range2_length - intersection_length
        # 计算IoU
        iou = intersection_length / union_length if union_length != 0 else 0 
        return iou

class TPGroup(Args):
    def __init__(
        self,
        ctxs: TrainerCtxs,
        node_idx: int,
        tp: int,
        devices: List[int],
        sr_list: List[float],
        unused: bool = False
    ):
        assert len(devices) == len(sr_list), "length mismatches"
        tp_devices_num = len(devices)
        self.ctxs = ctxs
        self.node_idx = node_idx
        self.tp = tp
        self.devices = devices
        self.sr_list = sr_list
        self.unused = unused
        
        self.hetero_ratio = self.tp // tp_devices_num
        self.alpha = ctxs.hetero_tp_alpha[int(math.log2(self.tp // tp_devices_num))]
        self.sr = max(sr_list) * self.alpha
        # 减少搜索空间
        if self.sr < ctxs.straggler_threshold:
            self.sr = 1.0
        
    def copy(self):
        return TPGroup(
            self.ctxs,
            self.node_idx,
            self.tp,
            self.devices,
            self.sr_list,
            self.unused 
        )
    
    @staticmethod    
    def print(item) -> float:
        if isinstance(item, TPGroup):
            return item.sr
        elif item is None:
            # 如果是 None，则返回一个字符串 'None'
            return 1.0
        else:
            raise ValueError(f"item type {type(item) is not supported}")

class StrategyModel:
    def __init__(
        self, 
        ctxs: TrainerCtxs,
        old_strategy_args: TrainerStrategyArgs, 
        used_devices_sr: Dict[int, float], 
        suspended_devices_sr: Dict[int, float],
        unused_devices: List[int]
    ):
        self.all_devices_num = old_strategy_args.dp * old_strategy_args.tp * old_strategy_args.pp
        self.all_nodes_num = self.all_devices_num // DEVICES_PER_NODE
        assert len(used_devices_sr) + len(suspended_devices_sr) + len(unused_devices) ==  self.all_devices_num, \
            "the sum of used devices and unused devices should be equal to all devices"
        self.ctxs = ctxs
        self.dp = old_strategy_args.dp
        self.tp = old_strategy_args.tp
        self.pp = old_strategy_args.pp
        self.zero = old_strategy_args.zero
        # 记录每个device当前所具有的layers情况
        # 后续会用来优化映射
        # 来减少热切换的代价（主要是显存代价）
        self.device_to_layers_prop = {}
        accumulate_ranks = 0
        for pipeline_idx in range(len(old_strategy_args.hetero_layers)):
            accumulate_layers = 0
            for stage_idx in range(len(old_strategy_args.hetero_layers[pipeline_idx])):
                start_layer_idx = accumulate_layers
                end_layer_idx = accumulate_layers + old_strategy_args.hetero_layers[pipeline_idx][stage_idx]
                start_rank_idx = accumulate_ranks
                end_rank_idx = accumulate_ranks + self.tp
                tp_devices = []
                for rank_idx in range(start_rank_idx, end_rank_idx):
                    assert rank_idx in old_strategy_args.rank_to_device_mapping, f"cannot find rank {rank_idx} in the old mapping"
                    device_idx = old_strategy_args.rank_to_device_mapping[rank_idx]
                    if rank_idx not in old_strategy_args.unused_rank_list:
                        tp_devices.append(device_idx)
                for slice_num, device_idx in enumerate(tp_devices):
                    assert device_idx not in self.device_to_layers_prop, f"duplicate device {device_idx}"
                    self.device_to_layers_prop[device_idx] = LayersProp(
                        start_layer_idx,
                        end_layer_idx,
                        slice_num,
                        len(tp_devices)
                    )
                    # print(device_idx, start_layer_idx, end_layer_idx, slice_num)
                accumulate_layers += old_strategy_args.hetero_layers[pipeline_idx][stage_idx]
                accumulate_ranks += self.tp
            assert accumulate_layers == self.ctxs.normal_layers * self.pp, "layers number mismatches"       
        self.used_devices_sr = used_devices_sr
        self.suspended_devices_sr = suspended_devices_sr
        self.unused_devices = unused_devices
        # Results
        self.strategies = None
        self.ds_parallel_configs = None
        used_devices_set = set(used_devices_sr.keys())
        suspended_devices_set = set(suspended_devices_sr.keys())
        unused_devices_set = set(unused_devices)
        assert bool(used_devices_set & suspended_devices_set) == False \
            and bool(used_devices_set & unused_devices_set) == False \
            and bool(suspended_devices_set & unused_devices_set) == False \
            , "different types of devices shouldn't have overlap"
        
    def __eq__(self, other):
        if not isinstance(other, StrategyModel):
            return False
        flag = (
            self.ctxs == other.ctxs and 
            self.dp == other.dp and 
            self.tp == other.tp and
            self.pp == other.pp and
            self.zero == other.zero and
            self.unused_devices == other.unused_devices
        )
        if flag == False:
            return flag_1
        # 对于used devices
        # 其使用compute stream进行profile
        # 比较精准
        for k, v in self.used_devices_sr.items():
            if k not in other.used_devices_sr:
                return False
            if abs(v - other.used_devices_sr[k]) >= self.ctxs.straggler_safe_gap:
                return False
        # 对于suspended devices
        # 其使用workload进行profile
        # 并不是很精准
        # 这里为了减少策略的频繁更换
        # 我们不区分被判定为是straggler的那些suspended devices的sr
        for k, v in self.suspended_devices_sr.items():
            if k not in other.suspended_devices_sr:
                return False
            if v >= self.ctxs.straggler_threshold and other.suspended_devices_sr[k] >= self.ctxs.straggler_threshold:
                pass
            elif abs(v - other.suspended_devices_sr[k]) >= self.ctxs.straggler_safe_gap:
                return False
        return True
     
    def make_plans(self):
        if self.strategies != None and self.ds_parallel_configs != None:
            return self.strategies, self.ds_parallel_configs
        # all_tp_groups, new_suspended_devices, new_unused_devices = self.solve_tp_arrangments()
        # support hetero pipeline stages
        clock = time.time()
        all_tp_groups, new_suspended_devices, new_unused_devices = self.solve_tp_arrangments_new()
        print("solve tp arrangement time =", time.time() - clock)
        clock = time.time()
        pipelines_list, l_values_list, m_values_list = self.enumerate_pp_pattern(all_tp_groups)
        print("solve pp arrangement time =", time.time() - clock)
        total_strategy_num = len(pipelines_list)
        self.strategies = []
        self.ds_parallel_configs = []
        for strategy_id in range(total_strategy_num):
            rank_to_device_mapping = {}
            suspended_rank_list = []
            no_layer_rank_list = []
            unused_rank_list = []
            pipelines = pipelines_list[strategy_id]
            hetero_stages = [len(pipelines[pipeline_idx]) for pipeline_idx in range(len(pipelines))]
            base_rank_idx = 0
            total_ranks = sum(hetero_stages) * self.tp
            dummy_device_idx = self.dp * self.tp * self.pp
            # 注意总的rank数可能大于总的device数
            # 引入一些dummy devices占位
            for pipeline_idx in range(self.dp):
                for stage_idx in range(hetero_stages[pipeline_idx]):
                    start_rank_idx = base_rank_idx + stage_idx * self.tp
                    for rank_idx in range(start_rank_idx, start_rank_idx + self.tp):
                        tp_size = len(pipelines[pipeline_idx][stage_idx].devices)
                        if rank_idx - start_rank_idx < tp_size:
                            # print(rank_idx, pipeline_idx, stage_idx, pipelines[pipeline_idx][stage_idx])
                            rank_to_device_mapping[rank_idx] = pipelines[pipeline_idx][stage_idx].devices[rank_idx - start_rank_idx]
                            if l_values_list[strategy_id][pipeline_idx][stage_idx] == 0:
                                no_layer_rank_list.append(rank_idx)
                        else:
                            if len(suspended_rank_list) < len(new_suspended_devices):
                                # 随便建立一个映射即可
                                rank_to_device_mapping[rank_idx] = new_suspended_devices[len(suspended_rank_list)]
                                suspended_rank_list.append(rank_idx)
                            elif len(unused_rank_list) < len(new_unused_devices):
                                # 随便建立一个映射即可
                                rank_to_device_mapping[rank_idx] = new_unused_devices[len(unused_rank_list)]
                                unused_rank_list.append(rank_idx)
                            elif dummy_device_idx < total_ranks:
                                # 建立一个dummy映射即可
                                rank_to_device_mapping[rank_idx] = dummy_device_idx
                                unused_rank_list.append(rank_idx)
                                dummy_device_idx += 1
                            else:
                                assert False, "unreachable"
                base_rank_idx += hetero_stages[pipeline_idx] * self.tp
            suspended_rank_list = suspended_rank_list + no_layer_rank_list
            strategy = TrainerStrategyArgs(
                dp=self.dp,
                tp=self.tp,
                pp=self.pp,
                zero=self.zero,
                rank_to_device_mapping=rank_to_device_mapping,
                suspended_rank_list=suspended_rank_list,
                unused_rank_list=unused_rank_list,
                hetero_data=True,
                hetero_layers=l_values_list[strategy_id],
                hetero_stages=hetero_stages,
                hetero_micro_batch_num_list=m_values_list[strategy_id]
            )
            accumulate_val = 0
            accumulate_hetero_stages = [0,]
            for val in hetero_stages:
                accumulate_val += val
                accumulate_hetero_stages.append(accumulate_val)
            ds_parallel_config = config_spread_zero(
                generate_gpt_hetero_4d_config(
                    "null",
                    rank_to_device_mapping, 
                    suspended_rank_list + unused_rank_list, 
                    l_values_list[strategy_id], 
                    accumulate_hetero_stages,
                    self.pp * self.ctxs.normal_layers, 
                    self.dp * self.tp * self.pp, 
                    self.dp, 
                    self.tp, 
                    self.pp, 
                    self.zero
                )
            )
            self.strategies.append(strategy)
            self.ds_parallel_configs.append(ds_parallel_config)
        return self.strategies, self.ds_parallel_configs
    
    # support tp group num != dp * pp
    def solve_tp_arrangments_new(self):
        available_devices_sr = self.used_devices_sr.copy()
        # 为了减少策略的频繁更换
        # 只有当suspended devices重新变回非straggler时
        # 我们才考虑将其纳入新的候选devices中
        available_devices_sr.update(
            {device_idx: sr for device_idx, sr in self.suspended_devices_sr.items() if sr < self.ctxs.straggler_threshold}
        )
        new_suspended_devices = [device_idx for device_idx, sr in self.suspended_devices_sr.items() if sr >= self.ctxs.straggler_threshold]
        # print(f"available_devices_sr = {available_devices_sr}")
        all_tp_groups = []
        # 对于所有node
        for node_idx in range(self.all_nodes_num):
            node_available_devices_num = 0
            node_devices_num = DEVICES_PER_NODE
            node_devices_range = range(node_idx * node_devices_num, (node_idx + 1) * node_devices_num)
            # 要么全在unused devices中
            # 要么一个都不在
            is_unused = [int(device_idx in self.unused_devices) for device_idx in node_devices_range]
            node_unused_num = sum(is_unused)
            assert node_unused_num == 0 or node_unused_num == node_devices_num, \
                "now only support all-or-nothing unused ranks in a typical node"
            if node_unused_num == node_devices_num:
                continue
            for device_idx in node_devices_range:
                if device_idx not in available_devices_sr:
                    continue
                node_available_devices_num += 1
            node_max_homo_devices_num = node_devices_num - self.tp
            assert node_available_devices_num > node_max_homo_devices_num, \
                f"{node_available_devices_num} <= {node_max_homo_devices_num}: unused ranks currently shouldn't be larger than tp - 1"
            node_available_devices_sr = {k: v for k, v in available_devices_sr.items() if k in node_devices_range}
            # 非straggler按照device的序号升序排列
            # straggler按照sr升序排
            # 枚举straggler部分的顺序
            node_available_devices_sr = sorted(
                node_available_devices_sr.items(), 
                key=lambda x: x[0] if x[1] < self.ctxs.straggler_threshold else x[1] * node_devices_num, 
            )
            straggler_num = len([device_idx for device_idx, sr in node_available_devices_sr if sr >= self.ctxs.straggler_threshold])
            best_hetero_tp_R_val = 0
            best_hetero_tp_list = []
            best_node_available_devices_sr = node_available_devices_sr.copy()
            basic_node_available_devices_sr = node_available_devices_sr.copy()
            # 找最优的tp划分
            def find_best_tp_split():
                nonlocal best_hetero_tp_R_val, best_hetero_tp_list, best_node_available_devices_sr
                hetero_tp_max = 1
                while hetero_tp_max <= node_available_devices_num - node_max_homo_devices_num:
                    hetero_tp_max *= 2
                hetero_tp_max = hetero_tp_max // 2
                # 寻找最优的hetero tp split方式
                begin_hetero_tp = hetero_tp_max
                start_idx = node_max_homo_devices_num - 1
                while begin_hetero_tp >= 1:
                    idx = start_idx
                    hetero_tp = begin_hetero_tp
                    hetero_tp_R_val = 0
                    hetero_tp_list = []
                    while hetero_tp >= 1:
                        idx = idx + hetero_tp
                        if idx > node_available_devices_num - 1:
                            break
                        hetero_tp_list.append(hetero_tp)
                        relative_hetero_idx = int(math.log2(self.tp // hetero_tp))
                        alpha = self.ctxs.hetero_tp_alpha[relative_hetero_idx]
                        weight = self.ctxs.hetero_tp_weight[relative_hetero_idx]
                        sr = max(device_sr[1] for device_sr in node_available_devices_sr[idx - hetero_tp + 1: idx + 1])
                        hetero_tp_R_val += 1 / (alpha * sr * weight)
                        hetero_tp = hetero_tp // 2
                    if hetero_tp_R_val > best_hetero_tp_R_val:
                        best_hetero_tp_R_val = hetero_tp_R_val
                        best_hetero_tp_list = hetero_tp_list.copy()
                        best_node_available_devices_sr = node_available_devices_sr.copy()
                    begin_hetero_tp = begin_hetero_tp // 2
            if straggler_num <= 1:
                find_best_tp_split()
            else:
                for straggler_seq_len in range(1, straggler_num):
                    node_available_devices_sr = basic_node_available_devices_sr.copy()
                    for enum, index in enumerate(reversed(range(-1, len(node_available_devices_sr) - straggler_num))):
                        # control tp split to run ablation experiments
                        '''
                        if straggler_seq_len != 2 or enum != 3:
                            if index != -1:
                                tmp = node_available_devices_sr.pop(index)
                                node_available_devices_sr.insert(index + straggler_seq_len, tmp)
                            continue
                        '''
                        find_best_tp_split()
                        print(f"straggler_seq_len {straggler_seq_len} enum {enum}, node_available_devices_sr =", node_available_devices_sr, "and best_hetero_tp_list =", best_hetero_tp_list, "and best_hetero_tp_R_val =", best_hetero_tp_R_val)
                        if index != -1:
                            tmp = node_available_devices_sr.pop(index)
                            node_available_devices_sr.insert(index + straggler_seq_len, tmp)
            print("best_node_available_devices_sr =", best_node_available_devices_sr, "and best_hetero_tp_list =", best_hetero_tp_list)
            final_used_devices = node_devices_num - self.tp + sum(best_hetero_tp_list)
            # print("final_used_devices =", final_used_devices, "best hetero tp list =", best_hetero_tp_list)
            # 常规tp组
            visited_devices = {}
            for tp_idx in range(node_max_homo_devices_num // self.tp):
                tp_devices = []
                tp_sr = []
                slice_num = 0
                while slice_num < self.tp:
                    find_flag = False
                    best_iou = -1
                    best_idx = 0
                    for idx in range(node_max_homo_devices_num):
                        device_idx = best_node_available_devices_sr[idx][0]
                        if device_idx not in visited_devices and \
                                self.device_to_layers_prop[device_idx].slice_num == slice_num and \
                                self.device_to_layers_prop[device_idx].slices_sum == self.tp:
                            # 对之后的slice num采用best fit
                            if slice_num >= 1:
                                first_slice_begin_l = self.device_to_layers_prop[tp_devices[0]].begin_layer_idx
                                first_slice_end_l = self.device_to_layers_prop[tp_devices[0]].end_layer_idx
                                begin_l = self.device_to_layers_prop[device_idx].begin_layer_idx
                                end_l = self.device_to_layers_prop[device_idx].end_layer_idx
                                # print(tp_devices[0], first_slice_begin_l, first_slice_end_l)
                                # print(device_idx, begin_l, end_l)
                                iou = LayersProp.calculate_iou((begin_l, end_l), (first_slice_begin_l, first_slice_end_l))
                                if iou > best_iou:
                                    best_iou = iou
                                    best_idx = idx
                                    find_flag = True
                            else:
                                visited_devices[device_idx] = True
                                tp_devices.append(device_idx)
                                tp_sr.append(best_node_available_devices_sr[idx][1])
                                find_flag = True
                                break
                    if not find_flag:
                        for idx in range(node_max_homo_devices_num):
                            device_idx = best_node_available_devices_sr[idx][0]
                            if device_idx not in visited_devices:
                                visited_devices[device_idx] = True
                                tp_devices.append(device_idx)
                                tp_sr.append(best_node_available_devices_sr[idx][1])
                                break
                    # 对之后的slice num采用best fit
                    elif slice_num >= 1:
                        visited_devices[best_node_available_devices_sr[best_idx][0]] = True
                        tp_devices.append(best_node_available_devices_sr[best_idx][0])
                        tp_sr.append(best_node_available_devices_sr[best_idx][1])
                    slice_num += 1
                all_tp_groups.append(
                    TPGroup(
                        self.ctxs, 
                        node_idx, 
                        self.tp, 
                        tp_devices, 
                        tp_sr
                    )
                ) 
            # 异构tp组
            start_hetero_idx = node_max_homo_devices_num
            for hetero_tp in best_hetero_tp_list:
                end_hetero_idx = start_hetero_idx + hetero_tp
                tp_devices = []
                tp_sr = []
                # TODO: find the best mapping
                # 这里先按顺序组tp了
                for idx in range(start_hetero_idx, end_hetero_idx):
                    tp_devices.append(best_node_available_devices_sr[idx][0])
                    tp_sr.append(best_node_available_devices_sr[idx][1])
                all_tp_groups.append(
                    TPGroup(
                        self.ctxs, 
                        node_idx, 
                        self.tp, 
                        tp_devices, 
                        tp_sr
                    )
                )   
                start_hetero_idx = end_hetero_idx
            # 求解新的suspended devices
            for idx in range(final_used_devices, node_available_devices_num):
                new_suspended_devices.append(best_node_available_devices_sr[idx][0])
        return all_tp_groups, new_suspended_devices, self.unused_devices
      
    # deprecated 
    def solve_tp_arrangments(self):
        available_devices_sr = self.used_devices_sr.copy()
        # 为了减少策略的频繁更换
        # 只有当suspended devices重新变回非straggler时
        # 我们才考虑将其纳入新的候选devices中
        available_devices_sr.update(
            {device_idx: sr for device_idx, sr in self.suspended_devices_sr.items() if sr < self.ctxs.straggler_threshold}
        )
        new_suspended_devices = [device_idx for device_idx, sr in self.suspended_devices_sr.items() if sr >= self.ctxs.straggler_threshold]
        # print(f"available_devices_sr = {available_devices_sr}")
        all_tp_groups = []
        # 对于所有node
        for node_idx in range(self.all_nodes_num):
            node_available_devices_num = 0
            node_devices_num = DEVICES_PER_NODE
            node_devices_range = range(node_idx * node_devices_num, (node_idx + 1) * node_devices_num)
            # 要么全在unused devices中
            # 要么一个都不在
            is_unused = [int(device_idx in self.unused_devices) for device_idx in node_devices_range]
            node_unused_num = sum(is_unused)
            assert node_unused_num == 0 or node_unused_num == node_devices_num, \
                "now only support all-or-nothing unused ranks in a typical node"
            if node_unused_num == node_devices_num:
                # 按顺序组建dummy tp group
                tp_cnt = 0
                tp_devices = []
                tp_sr = []
                for device_idx in node_devices_range:
                    tp_cnt += 1
                    tp_devices.append(device_idx)
                    tp_sr.append(INF_SR)
                    if tp_cnt == self.tp:
                        all_tp_groups.append(
                            TPGroup(
                                self.ctxs, 
                                node_idx, 
                                self.tp, 
                                tp_devices, 
                                tp_sr, 
                                unused=True
                            )
                        )    
                        tp_devices.clear()
                        tp_sr.clear()
                        tp_cnt = 0
                continue
            for device_idx in node_devices_range:
                if device_idx not in available_devices_sr:
                    continue
                node_available_devices_num += 1
            node_max_homo_devices_num = node_devices_num - self.tp
            assert node_available_devices_num > node_max_homo_devices_num, \
                f"{node_available_devices_num} <= {node_max_homo_devices_num}: unused ranks currently shouldn't be larger than tp - 1"
            node_available_devices_sr = {k: v for k, v in available_devices_sr.items() if k in node_devices_range}
            # 非straggler按照device的序号升序排列
            # straggler按照sr升序排
            node_available_devices_sr = sorted(
                node_available_devices_sr.items(), 
                key=lambda x: x[0] if x[1] < self.ctxs.straggler_threshold else x[1] * node_devices_num, 
            )
            print("node_available_devices_sr =", node_available_devices_sr)
            hetero_tp = 1
            start_idx = node_max_homo_devices_num - 1
            hetero_tp_val = []
            while True:
                idx = start_idx + hetero_tp
                if idx > node_available_devices_num - 1:
                    break
                relative_hetero_idx = int(math.log2(self.tp // hetero_tp))
                alpha = self.ctxs.hetero_tp_alpha[relative_hetero_idx]
                weight = self.ctxs.hetero_tp_weight[relative_hetero_idx]
                sr = node_available_devices_sr[idx][1]
                hetero_tp_val.append(alpha * sr * weight)
                hetero_tp *= 2
            best_hetero_tp_val = min(hetero_tp_val)
            final_relative_hetero_idx = hetero_tp_val.index(best_hetero_tp_val)
            final_hetero_tp = 2 ** final_relative_hetero_idx
            final_used_devices = node_devices_num - self.tp + final_hetero_tp
            # print("hetero_tp_val =", hetero_tp_val, "final_used_devices =", final_used_devices)
            # 求解tp组
            for tp_idx in range(math.ceil(final_used_devices / self.tp)):
                tp_devices = []
                tp_sr = []
                for idx in range(tp_idx * self.tp, min((tp_idx + 1) * self.tp, final_used_devices)):
                    tp_devices.append(node_available_devices_sr[idx][0])
                    tp_sr.append(node_available_devices_sr[idx][1])
                all_tp_groups.append(
                    TPGroup(
                        self.ctxs, 
                        node_idx, 
                        self.tp, 
                        tp_devices, 
                        tp_sr
                    )
                )    
            # 求解新的suspended devices
            for idx in range(final_used_devices, node_available_devices_num):
                new_suspended_devices.append(node_available_devices_sr[idx][0])
        return all_tp_groups, new_suspended_devices, self.unused_devices
    
    def enumerate_pp_pattern(self, all_tp_groups: List[TPGroup]):
        # 启发式
        # 大部分tp是正常的
        # straggler tp groups按straggler ratio从大到小排
        # Args.print_args_list(all_tp_groups)     
        straggler_tp_groups = [tp_group for tp_group in all_tp_groups if tp_group.sr > 1.0]
        straggler_tp_groups.sort(
            key=lambda x: x.sr, 
            reverse=True
        )
        # normal tp groups按node分后再按device编号从小到大排
        normal_tp_groups_mapping = {
            node_idx: [tp_group for tp_group in all_tp_groups if tp_group.sr == 1.0 and tp_group.node_idx == node_idx]
                for node_idx in range(self.all_nodes_num)
        }
        total_normal_tp_group_num_mapping = {
            node_idx: len(normal_tp_groups_mapping[node_idx])
                for node_idx in range(self.all_nodes_num)
        }
        for node_idx in range(self.all_nodes_num):
            normal_tp_groups_mapping[node_idx].sort(
                key=lambda x: max(x.devices)
            )
            
        # step 0
        # workaround
        # 非必要不异构stage数目
        hetero_stages_plans = []
        if self.dp * self.pp == len(all_tp_groups):
            hetero_stages = [self.pp for _ in range(self.dp)]
            hetero_stages_plans.append(hetero_stages)
        else:
            # heuristic
            # 目前策略是尽量让stage数平均
            base_stages = len(all_tp_groups) // self.dp
            remain_stages = len(all_tp_groups) - self.dp * base_stages
            search_stage_gap = 1
            '''
            hetero_stages_plan_1 = [base_stages + remain_stages if idx == 0 else base_stages for idx in range(self.dp)]
            hetero_stages_plans.append(hetero_stages_plan_1)
            hetero_stages_plan_2 = [base_stages + 1 if idx < remain_stages else base_stages for idx in range(self.dp)]
            hetero_stages_plans.append(hetero_stages_plan_2)
            '''
            hetero_stages_plans = []
            hetero_stages_plan = []
            all_stages = 0
            def search_hetero_stage_plan(pipeline_idx):
                nonlocal hetero_stages_plans, hetero_stages_plan, all_stages
                if pipeline_idx == self.dp:
                    if all_stages == len(all_tp_groups):
                        hetero_stages_plans.append(hetero_stages_plan.copy())
                    return
                for stage_num in range(base_stages - search_stage_gap, base_stages + remain_stages + search_stage_gap + 1):
                    if pipeline_idx >= 1 and stage_num > hetero_stages_plan[pipeline_idx - 1]:
                        continue
                    hetero_stages_plan.append(stage_num)
                    all_stages += stage_num
                    search_hetero_stage_plan(pipeline_idx + 1)
                    hetero_stages_plan.pop()
                    all_stages -= stage_num
            search_hetero_stage_plan(0) 
            print("hetero_stages_plans =", hetero_stages_plans)          

        # step 1
        # 先把straggler tp group的位置枚举出来
        # 其余都用None代替
        all_pipelines_template = []
        pipelines = []
        pipelines_straggler_tp_group_num = []
        total_straggler_tp_group_num = len(straggler_tp_groups)
        total_normal_tp_group_num = sum(total_normal_tp_group_num_mapping.values())
        visited_straggler_tp_group = {}
        visited_normal_tp_group_num = 0
        # 启发式搜素
        # 所有pipeline的straggler数目从大到小排
        # 每个pipeline内部straggler tp group都放在前头的stage并从大到小排
        # 且要对pipeline permuation不变的进行剪枝
        # 否则搜索空间会溢出
        def dfs(hetero_stages_plan_idx, pipeline_idx):
            nonlocal all_pipelines_template, pipelines
            if pipeline_idx == self.dp:
                all_pipelines_template.append(pipelines.copy())
                return 
            pipeline = []
            pipeline_straggler_tp_group_num = 0
            def pipeline_dfs(
                stage_idx, 
                min_straggler_tp_group_idx, 
            ):
                nonlocal pipeline, pipeline_straggler_tp_group_num, \
                    pipelines, pipelines_straggler_tp_group_num, \
                    visited_straggler_tp_group, visited_normal_tp_group_num, \
                    hetero_stages_plans
                # 减枝
                # 靠后的pipeline不能比靠前的pipeline有更多的straggler tp group
                # 如果数目相等则靠后的pipeline开头的straggler tp group的sr要小于等于前者
                if pipeline_idx != 0 and pipeline_straggler_tp_group_num > pipelines_straggler_tp_group_num[pipeline_idx - 1]:
                    return
                if pipeline_idx != 0 and pipeline_straggler_tp_group_num >= 1 and pipeline_straggler_tp_group_num == pipelines_straggler_tp_group_num[pipeline_idx - 1]:
                    if pipeline[0].sr > pipelines[pipeline_idx - 1][0].sr:
                        return
                # 搜完一条pipeline接着搜下一条
                if stage_idx == hetero_stages_plans[hetero_stages_plan_idx][pipeline_idx]:
                    # 目前不支持整条pipeline都unused
                    # 即减少了dp维度
                    # 2024.6.9 Update
                    # 目前unused属性deprecated
                    # 通过hetero_stages_plans来控制pipeline长度
                    if pipeline[-1] != None and pipeline[-1].unused == True:
                        return
                    pipelines.append(pipeline.copy())
                    pipelines_straggler_tp_group_num.append(pipeline_straggler_tp_group_num)
                    # print(f"dfs pipeline {pipeline_idx} finished, visited_normal_tp_group_num = {visited_normal_tp_group_num}")
                    dfs(hetero_stages_plan_idx, pipeline_idx + 1)  
                    pipelines.pop()
                    pipelines_straggler_tp_group_num.pop()
                    return 
                # 从大到小选straggler tp group
                for straggler_tp_group_idx in range(min_straggler_tp_group_idx, total_straggler_tp_group_num):
                    if straggler_tp_group_idx in visited_straggler_tp_group:
                        continue
                    visited_straggler_tp_group[straggler_tp_group_idx] = True
                    pipeline_straggler_tp_group_num += 1
                    pipeline.append(straggler_tp_groups[straggler_tp_group_idx])
                    # pipeline的下一个stage的straggler ratio需要更小
                    pipeline_dfs(stage_idx + 1, straggler_tp_group_idx + 1)
                    pipeline.pop()
                    pipeline_straggler_tp_group_num -= 1
                    del visited_straggler_tp_group[straggler_tp_group_idx]
                # 不选straggler tp group
                if visited_normal_tp_group_num >= total_normal_tp_group_num:
                    return
                pipeline.append(None)
                visited_normal_tp_group_num += 1
                # 之后必须都选None
                pipeline_dfs(stage_idx + 1, total_straggler_tp_group_num)
                visited_normal_tp_group_num -= 1
                pipeline.pop()
            # 从stage 0开始搜该条pipeline
            pipeline_dfs(0, 0) 
        for hetero_stages_plan_idx in range(len(hetero_stages_plans)):
            # 对于所有hetero stages plan都从pipeline 0开始搜一遍
            dfs(hetero_stages_plan_idx, 0)    
        
        # step 2
        # 先筛选一波
        # 对于每一种pipeline template
        # 计算理想最优解
        assert len(all_pipelines_template) > 0, "something wrong, no available strategy"
        objective_list = []
        l_values_list = []
        m_values_list = []
        legal_pipelines_template = []
        for pipelines_template in all_pipelines_template:
            print_formatted_pipelines_template = [
                [TPGroup.print(item) for item in pipeline]
                    for pipeline in pipelines_template
            ]
            y_values = [
                [(tp_group.sr if tp_group != None else 1.0) for tp_group in pipeline]
                    for pipeline in pipelines_template
            ]
            hetero_values = [
                [(tp_group.hetero_ratio if tp_group != None else 1.0) for tp_group in pipeline]
                    for pipeline in pipelines_template
            ]
            is_unused = [
                [(tp_group.unused if tp_group != None else False) for tp_group in pipeline]
                    for pipeline in pipelines_template
            ]
            is_solved, objective, l_values, m_values = self.solve_pp_arrangement(y_values, hetero_values, is_unused)
            if not is_solved:
                continue
            objective_list.append(objective)
            l_values_list.append(l_values)
            m_values_list.append(m_values)
            legal_pipelines_template.append(pipelines_template)
        top_k = min(self.ctxs.top_k, len(objective_list))       
        assert top_k != 0, "no possible strategies"
        top_k_idx = heapq.nsmallest(top_k, range(len(objective_list)), key=lambda i: objective_list[i])
        top_k_objective = [objective_list[i] for i in top_k_idx]
        top_k_l_values = [l_values_list[i] for i in top_k_idx]
        top_k_m_values = [m_values_list[i] for i in top_k_idx]
        top_k_pipelines_template = [
            [
                [
                    legal_pipeline_stage_template.copy() if legal_pipeline_stage_template != None else None 
                        for legal_pipeline_stage_template in legal_pipeline_template
                ] 
                for legal_pipeline_template in legal_pipelines_template[i]
            ] 
            for i in top_k_idx
        ]
        
        # log
        for i in range(top_k):
            print_formatted_pipelines_template = [
                [TPGroup.print(item) for item in pipeline]
                    for pipeline in top_k_pipelines_template[i]
            ]
            print_formatted_pipelines_template_extra = [
                [item.devices if item != None else None for item in pipeline]
                    for pipeline in top_k_pipelines_template[i]
            ]
            print("**********************************************") 
            print("pipelines_template =", print_formatted_pipelines_template)
            print("pipelines_template_devices =", print_formatted_pipelines_template_extra)
            print("objective =", top_k_objective[i])
            print("l_values =", top_k_l_values[i])
            print("m_values =", top_k_m_values[i])
        
        # step 3
        # 对于top k的pipeline template再依次分配每个normal tp group
        # 采用启发式策略
        # 从stage靠前的地方往后枚举
        # 尽量让跨node的grad reduce尽量少
        # 当不得不存在跨node通信时
        # 尽量让node选取和最慢的straggler一致
        for top_idx, (pipelines_template, l_lists) in enumerate(zip(top_k_pipelines_template, top_k_l_values)):
            # print(top_k_pipelines_template)
            accumulate_l_lists = []
            for l_list in l_lists:
                accumulate_l_list = [0,]
                accumulate_l = 0
                for l in l_list:
                    accumulate_l += l
                    accumulate_l_list.append(accumulate_l)
                accumulate_l_lists.append(accumulate_l_list)
            # print(pipelines_template, l_lists, accumulate_l_lists)
            visited_normal_tp_groups_mapping = [
                {} for node_idx in range(self.all_nodes_num)
            ]
            # step 3.1
            # 第一遍pass
            # 先尽量减少跨机grad reduce
            # 把能填的None填了
            max_hetero_stages = len(pipelines_template[0])
            for stage_id in range(max_hetero_stages):
                if pipelines_template[0][stage_id] == None:
                    continue
                suggested_node_idx = pipelines_template[0][stage_id].node_idx
                for pipeline_id in range(self.dp):
                    if stage_id >= len(pipelines_template[pipeline_id]):
                        continue
                    if pipelines_template[pipeline_id][stage_id] == None:
                        # layers range IoU best fit strategy
                        begin_l = accumulate_l_lists[pipeline_id][stage_id]
                        end_l = accumulate_l_lists[pipeline_id][stage_id + 1]
                        best_iou = -1
                        best_num = 0
                        for curr_num in range(total_normal_tp_group_num_mapping[suggested_node_idx]):
                            if curr_num in visited_normal_tp_groups_mapping[suggested_node_idx]:
                                continue
                            tp_group = normal_tp_groups_mapping[suggested_node_idx][curr_num]
                            iou = 0
                            for device_idx in tp_group.devices:
                                old_begin_l = self.device_to_layers_prop[device_idx].begin_layer_idx
                                old_end_l = self.device_to_layers_prop[device_idx].end_layer_idx
                                iou += LayersProp.calculate_iou((begin_l, end_l), (old_begin_l, old_end_l))
                            if iou > best_iou:
                                best_iou = iou
                                best_num = curr_num
                        if best_iou != -1:
                            pipelines_template[pipeline_id][stage_id] = normal_tp_groups_mapping[suggested_node_idx][best_num]
                            visited_normal_tp_groups_mapping[suggested_node_idx][best_num] = True
                            # print(f"pipeline id {pipeline_id} stage id {stage_id}, normal tp group is {normal_tp_groups_mapping[suggested_node_idx][best_num]}")
            # step 3.2
            # 第二遍pass
            # 把剩下的None按顺序填了
            for stage_id in range(max_hetero_stages):
                for pipeline_id in range(self.dp):
                    if stage_id >= len(pipelines_template[pipeline_id]):
                        continue
                    if pipelines_template[pipeline_id][stage_id] == None:
                        replace_none = False
                        # layers range IoU best fit strategy
                        begin_l = accumulate_l_lists[pipeline_id][stage_id]
                        end_l = accumulate_l_lists[pipeline_id][stage_id + 1]
                        best_iou = -1
                        best_node_idx = 0
                        best_num = 0
                        for node_idx in range(self.all_nodes_num):
                            for curr_num in range(total_normal_tp_group_num_mapping[node_idx]):
                                if curr_num in visited_normal_tp_groups_mapping[node_idx]:
                                    continue
                                tp_group = normal_tp_groups_mapping[node_idx][curr_num]
                                iou = 0
                                for device_idx in tp_group.devices:
                                    old_begin_l = self.device_to_layers_prop[device_idx].begin_layer_idx
                                    old_end_l = self.device_to_layers_prop[device_idx].end_layer_idx
                                    iou += LayersProp.calculate_iou((begin_l, end_l), (old_begin_l, old_end_l))
                                if iou > best_iou:
                                    best_iou = iou
                                    best_node_idx = node_idx
                                    best_num = curr_num
                                    replace_none = True
                        assert replace_none == True, "can't find a normal tp group to place here"
                        pipelines_template[pipeline_id][stage_id] = normal_tp_groups_mapping[best_node_idx][best_num]
                        visited_normal_tp_groups_mapping[best_node_idx][best_num] = True
                        # print(f"pipeline id {pipeline_id} stage id {stage_id}, normal tp group is {normal_tp_groups_mapping[best_node_idx][best_num]}")
                        # print(visited_normal_tp_groups_mapping)
            # top_k_pipelines_template[top_idx] = pipelines_template
        
        # 最终返回top k个搜索出的策略
        return top_k_pipelines_template, top_k_l_values, top_k_m_values
                                                            
    def solve_pp_arrangement(
        self, 
        y_values: List[List[float]], 
        hetero_values: List[List[float]],
        is_unused: List[List[bool]],
        only_adjust_batch: bool = False
    ):    
        # print(y_values, hetero_values, is_unused)
        clock = time.time()
        
        # 第一个整数线性规划问题
        dp = self.dp
        pp = self.pp
        L = self.pp * self.ctxs.normal_layers
        C = self.ctxs.memory_bound - self.ctxs.memory_safe_gap
        k_values = self.ctxs.memory_k
        embedding_value = self.ctxs.memory_embedding
        extra_value = self.ctxs.memory_extra
        # d_values = self.ctxs.memory_d
        # 最优解的值
        o_values = [] 
        l_values = []
        # 第二个整数线性规划问题
        B_div_b = self.dp * self.ctxs.normal_mbn
        # 最优解的值
        m_values = [] 
        
        pulp.LpSolverDefault.msg = 0
        if only_adjust_batch:
            for i in range(dp):
                l_i_values = []
                max_y_times_l = 0
                hetero_stages = len(y_values[i])
                for j in range(hetero_stages):
                    if j == hetero_stages - 1:
                        l = L - (hetero_stages - 1) * (L // hetero_stages)
                    else:
                        l = L // hetero_stages
                    l_i_values.append(l)
                    max_y_times_l = max(max_y_times_l, y_values[i][j] * l)
                l_values.append(l_i_values)
                o_values.append(max_y_times_l)
        else:
            # 求解第一个整数线性规划问题
            for i in range(dp):
                hetero_stages = len(y_values[i])
                # 定义整数线性规划问题
                model = LpProblem(name="Integer_Linear_Programming_Problem_1", sense=LpMinimize)
                # 变量
                l_vars = [LpVariable(f"l_{i}_{j}", lowBound=0, cat="Integer") for j in range(hetero_stages)]
                max_y_times_l = LpVariable("max_y_times_l", lowBound=0, cat="Continuous")  # 最大值变量
                # 添加目标函数
                model += max_y_times_l
                # 约束条件
                model += lpSum(l_vars) == L
                for j in range(hetero_stages):
                    if is_unused[i][j]:
                        model += l_vars[j] == 0
                    else:
                        model += max_y_times_l >= y_values[i][j] * l_vars[j]
                        model += (k_values[-(hetero_stages - j)] * l_vars[j] + (embedding_value if j == 0 or j == hetero_stages - 1 else 0)) * hetero_values[i][j] + extra_value <= C
                # 求解问题
                model.solve()
                if LpStatus[model.status] != 'Optimal':
                    # print("No solution")
                    return False, None, None, None
                # 打印结果
                # print(f"Q1, num = {i}, Optimal Objective Value:", model.objective.value())
                o_values.append(model.objective.value())
                l_i_values = []
                for var in model.variables():
                    # print(var.name, "=", var.varValue)
                    if 'l_' in var.name:
                        l_i_values.append(int(var.varValue))
                l_values.append(l_i_values)
            
        # 求解第二个整数线性规划问题
        # 定义整数线性规划问题
        model = LpProblem(name="Integer_Linear_Programming_Problem_2", sense=LpMinimize)
        # 变量
        m_vars = [LpVariable(f"m_{i}", lowBound=0, cat="Integer") for i in range(dp)]
        max_o_times_m = LpVariable("max_o_times_m", lowBound=0, cat="Continuous")  # 最大值变量
        # 添加目标函数
        model += max_o_times_m
        # 约束条件
        model += lpSum(m_vars) == B_div_b
        for i in range(dp):
            model += max_o_times_m >= o_values[i] * m_vars[i]
            model += m_vars[i] >= len(y_values[i])
        # 求解问题
        model.solve()
        if LpStatus[model.status] != 'Optimal':
            print("No solution")
            return False, None, None, None
        # 打印结果
        # print("Q2, Optimal Objective Value:", model.objective.value())
        for var in model.variables():
            # print(var.name, "=", var.varValue)
            if 'm_' in var.name:
                m_values.append(int(var.varValue))

        # print("objective =", model.objective.value()) 
        # print("l_values =", l_values)  
        # print("m_values =", m_values) 
        # print("time elapse =", time.time() - clock) 
        return True, model.objective.value(), l_values, m_values
                
                        
                    
                
        