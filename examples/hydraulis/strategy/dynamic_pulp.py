import pulp
import time
import json
import random
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from pyscipopt import Model
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List
from .cost_model import dynamic_strategy_time_cost

MAX_WORKERS = 1
TRIAL_WORKERS = 1
MINI_TRIAL_NUM = 50

def process_trial(trial, batch_seqlen_array, strategy_pool, sorted_max_seqlen_list,
                  sorted_strategy_id_list, sorted_cur_strategy_relative_id):
    DP = len(sorted_strategy_id_list)
    original_indices = np.arange(len(batch_seqlen_array))
    best_accumulate_cost = float('inf')
    best_res = None
    pp_list = [strategy_pool['strategies'][sid]['pp'] for sid in sorted_strategy_id_list]
    
    # 缓存dynamic_strategy_time_cost的计算结果
    @lru_cache(maxsize=None)
    def cached_dynamic_strategy_time_cost(strategy_id, seqlen):
        return dynamic_strategy_time_cost(strategy_pool, strategy_id, seqlen)
        
    for mini_trial in range(trial, trial + MINI_TRIAL_NUM):
        random.seed(mini_trial)
        shuffled_indices = original_indices.copy()
        random.shuffle(shuffled_indices)
        shuffled_batch_seqlen_array = batch_seqlen_array[shuffled_indices]
        accumulate_cost = np.zeros(DP)
        pipeline_warmup_cooldown_cost = np.zeros(DP)
        total_cost = np.zeros(DP)
        res = {dp_id: [] for dp_id in range(DP)}
        # 初始分配：只能放在特定策略中的序列
        visited = set()
        for shuffled_seq_id, seqlen in enumerate(shuffled_batch_seqlen_array):
            # 判断是否只能放在dp_id=0的策略中
            if len(sorted_max_seqlen_list) == 1 or (sorted_max_seqlen_list[0] >= seqlen and sorted_max_seqlen_list[1] < seqlen):
                strategy_id = sorted_strategy_id_list[0]
                cost = cached_dynamic_strategy_time_cost(strategy_id, seqlen)
                accumulate_cost[0] += cost
                pp = pp_list[0]
                pipeline_warmup_cooldown_cost[0] = max(pipeline_warmup_cooldown_cost[0], cost * (pp - 1))
                total_cost[0] = accumulate_cost[0] + pipeline_warmup_cooldown_cost[0]
                res[0].append(shuffled_indices[shuffled_seq_id])
                visited.add(shuffled_seq_id)
        # 维护当前的最大累计开销
        max_total_cost = total_cost.max()
        # 分配其余的序列
        for shuffled_seq_id, seqlen in enumerate(shuffled_batch_seqlen_array):
            if shuffled_seq_id in visited:
                continue
            select_dp_id = None
            min_candidate_max_total_cost = float('inf')
            for dp_id in range(DP):
                if sorted_max_seqlen_list[dp_id] < seqlen:
                    continue  # 该策略无法容纳此序列长度
                strategy_id = sorted_strategy_id_list[dp_id]
                cost = cached_dynamic_strategy_time_cost(strategy_id, seqlen)
                pp = pp_list[dp_id]
                new_accumulate_cost = accumulate_cost[dp_id] + cost
                new_pipeline_cost = max(pipeline_warmup_cooldown_cost[dp_id], cost * (pp - 1))
                new_total_cost_dp_id = new_accumulate_cost + new_pipeline_cost
                # 计算新的最大累计开销
                candidate_max_total_cost = max(max_total_cost, new_total_cost_dp_id)
                if candidate_max_total_cost < min_candidate_max_total_cost:
                    min_candidate_max_total_cost = candidate_max_total_cost
                    select_dp_id = dp_id
                if candidate_max_total_cost == min_candidate_max_total_cost and accumulate_cost[dp_id] == 0:
                    min_candidate_max_total_cost = candidate_max_total_cost
                    select_dp_id = dp_id
            assert select_dp_id is not None, f"Cannot select a proper dp to place a sequence with length {seqlen}"
            # 更新选定策略的累计开销和结果
            strategy_id = sorted_strategy_id_list[select_dp_id]
            cost = cached_dynamic_strategy_time_cost(strategy_id, seqlen)
            pp = pp_list[select_dp_id]
            accumulate_cost[select_dp_id] += cost
            pipeline_warmup_cooldown_cost[select_dp_id] = max(pipeline_warmup_cooldown_cost[select_dp_id], cost * (pp -1))
            total_cost[select_dp_id] = accumulate_cost[select_dp_id] + pipeline_warmup_cooldown_cost[select_dp_id]
            max_total_cost = max(max_total_cost, total_cost[select_dp_id])
            res[select_dp_id].append(shuffled_indices[shuffled_seq_id])
        # 检查并更新最优结果
        if max_total_cost < best_accumulate_cost:
            best_accumulate_cost = max_total_cost
            best_res = res.copy()
        # 检查是否有未分配序列的策略
        has_empty = any(len(best_res[dp_id]) == 0 for dp_id in range(DP))
        if has_empty:
            return (float('inf'), None)
    return (best_accumulate_cost, best_res[sorted_cur_strategy_relative_id])

def process_trial_deprecated(trial, batch_seqlen_array, strategy_pool, sorted_max_seqlen_list, sorted_strategy_id_list, sorted_cur_strategy_relative_id):
    DP = len(sorted_strategy_id_list)
    # 取某策略支持的最大seqlen和当前batch中最大seqlen作为warmup-cooldown参考值
    # 已改为动态调整
    '''
    pipeline_warmup_cooldown_cost = np.array([
        (strategy_pool['strategies'][sorted_strategy_id_list[dp_id]]['pp'] - 1) * 
        dynamic_strategy_time_cost(strategy_pool, sorted_strategy_id_list[dp_id], min(sorted_max_seqlen_list[dp_id], batch_seqlen_array.max())) 
        for dp_id in range(DP)
    ])
    '''
    original_indices = list(range(len(batch_seqlen_array)))  # 原始的索引顺序
    best_accumulate_cost = float('inf')  # 记录最优的最大累积开销
    best_res = None  # 记录最优的分配结果
    for mini_trial in tqdm(range(trial, trial + MINI_TRIAL_NUM)):
        random.seed(mini_trial)
        shuffled_indices = original_indices.copy()
        random.shuffle(shuffled_indices)
        shuffled_batch_seqlen_array = batch_seqlen_array[shuffled_indices]
        accumulate_cost = np.array([0 for _ in range(DP)])
        pipeline_warmup_cooldown_cost = np.array([0 for _ in range(DP)])
        res = {}
        for dp_id in range(DP):
            res[dp_id] = []
        # 先扫一遍
        # 有些long seq只能放到其中某一种固定策略中
        visited = {}
        for shuffled_seq_id, seqlen in enumerate(shuffled_batch_seqlen_array):
            if sorted_max_seqlen_list[0] >= seqlen and sorted_max_seqlen_list[1] < seqlen:
                cost = dynamic_strategy_time_cost(strategy_pool, sorted_strategy_id_list[0], seqlen)
                accumulate_cost[0] += cost
                pipeline_warmup_cooldown_cost[0] = max(pipeline_warmup_cooldown_cost[0], cost * (strategy_pool['strategies'][sorted_strategy_id_list[0]]['pp'] - 1))
                res[0].append(shuffled_indices[shuffled_seq_id])
                visited[shuffled_seq_id] = True
        # 分配其余seq到当前累积开销最小的dp中
        for shuffled_seq_id, seqlen in enumerate(shuffled_batch_seqlen_array):
            if shuffled_seq_id in visited:
                continue
            select_dp_id = None
            min_max_accumulate_cost = float('inf')
            min_empty_dp = DP
            for dp_id in range(DP):
                if sorted_max_seqlen_list[dp_id] < seqlen:
                    break
                cost = dynamic_strategy_time_cost(strategy_pool, sorted_strategy_id_list[dp_id], seqlen)
                old_pipeline_warmup_cooldown_cost = pipeline_warmup_cooldown_cost[dp_id]
                accumulate_cost[dp_id] += cost
                pipeline_warmup_cooldown_cost[dp_id] = max(pipeline_warmup_cooldown_cost[dp_id], cost * (strategy_pool['strategies'][sorted_strategy_id_list[dp_id]]['pp'] - 1))
                max_accumulate_cost = (accumulate_cost + pipeline_warmup_cooldown_cost).max()
                empty_dp = (np.array(accumulate_cost) == 0).sum()
                accumulate_cost[dp_id] -= cost
                pipeline_warmup_cooldown_cost[dp_id] = old_pipeline_warmup_cooldown_cost
                if max_accumulate_cost < min_max_accumulate_cost:
                    select_dp_id = dp_id
                    min_max_accumulate_cost = max_accumulate_cost
                    min_empty_dp = empty_dp
                elif max_accumulate_cost == min_max_accumulate_cost:
                    if empty_dp < min_empty_dp:
                        select_dp_id = dp_id
                        min_empty_dp = empty_dp
            assert select_dp_id is not None, f"Cannot select a proper dp to place a sequence with length {seqlen}"
            cost = dynamic_strategy_time_cost(strategy_pool, sorted_strategy_id_list[select_dp_id], seqlen)
            accumulate_cost[select_dp_id] += cost
            pipeline_warmup_cooldown_cost[dp_id] = max(pipeline_warmup_cooldown_cost[dp_id], cost * (strategy_pool['strategies'][sorted_strategy_id_list[dp_id]]['pp'] - 1))
            res[select_dp_id].append(shuffled_indices[shuffled_seq_id])
        # assert len(res) > 0, "Currently not support zero seqs"
        # 返回当前尝试的最大累积开销和结果
        current_max_accumulate_cost = (accumulate_cost + pipeline_warmup_cooldown_cost).max()
        if current_max_accumulate_cost < best_accumulate_cost:
            best_accumulate_cost = current_max_accumulate_cost
            best_res = res
    # print(f"best assigning results: {best_res}")
    has_empty = False
    for dp_id in range(DP):
        if len(best_res[dp_id]) == 0:
            has_empty = True
    if has_empty:
        return (float('inf'), None)
    return (best_accumulate_cost, best_res[sorted_cur_strategy_relative_id])

# 返回batch_seqlen_array中应该归属于当前strategy的indices
# 保证返回的indices按照从小到大的顺序排列
def dynamic_strategy(strategy_pool, strategy_id_list: List[int], max_seqlen_list: List[int], cur_strategy_relative_id: int, batch_seqlen_array: np.ndarray):
    # 先按照max_seqlen从大到小的顺序对strategy排序
    indices = list(range(len(strategy_id_list)))
    sorted_indices = sorted(indices, key=lambda x: max_seqlen_list[x], reverse=True)
    sorted_cur_strategy_relative_id = sorted_indices.index(cur_strategy_relative_id)
    sorted_strategy_id_list = [strategy_id_list[i] for i in sorted_indices]
    sorted_max_seqlen_list = [max_seqlen_list[i] for i in sorted_indices]
    # print(f"Dynamic strategy, sorted_strategy_id_list is {sorted_strategy_id_list}, sorted_cur_strategy_relative_id is {sorted_cur_strategy_relative_id}, sorted_max_seqlen_list is {max_seqlen_list}, considering batch_seqlen_array {batch_seqlen_array.tolist()}")
    # 线性规划输入数据
    assert len(batch_seqlen_array.shape) == 1, "sorted_len shape must be [global_batch_size,]"
    
    use_random_greedy_algorithm = True
    if use_random_greedy_algorithm:
        best_accumulate_cost = float('inf')  # 记录最优的最大累积开销
        best_res = None  # 记录最优的分配结果
        start_time = time.time()
        if TRIAL_WORKERS == 1:
            best_accumulate_cost, best_res = process_trial(
                0,
                batch_seqlen_array,
                strategy_pool,
                sorted_max_seqlen_list,
                sorted_strategy_id_list,
                sorted_cur_strategy_relative_id
            )
        else:
            with ProcessPoolExecutor(max_workers=TRIAL_WORKERS) as executor:
                # 并行执行process_trial
                results = executor.map(
                    process_trial,
                    range(0, TRIAL_WORKERS * MINI_TRIAL_NUM, MINI_TRIAL_NUM),
                    [batch_seqlen_array] * TRIAL_WORKERS,
                    [strategy_pool] * TRIAL_WORKERS,
                    [sorted_max_seqlen_list] * TRIAL_WORKERS,
                    [sorted_strategy_id_list] * TRIAL_WORKERS,
                    [sorted_cur_strategy_relative_id] * TRIAL_WORKERS
                )
                # 收集结果
                for current_max_accumulate_cost, res in results:
                    # 如果当前尝试的最大累积开销比之前的最优解小，则更新最优解
                    if current_max_accumulate_cost < best_accumulate_cost:
                        best_accumulate_cost = current_max_accumulate_cost
                        best_res = res
        end_time = time.time()
        print(f"assign seqs time cost is {end_time - start_time}s")
        if best_res is not None:
            best_res = sorted(best_res)
        # 返回最优解
        return (best_accumulate_cost, best_res)
                
    B = len(batch_seqlen_array)  # 序列条数
    DP = len(sorted_strategy_id_list)  # 数据并行个数
    S = batch_seqlen_array  # 序列长度
    J = []  # 每个序列所能达到的max_seqlen最小的策略编号
    for i in range(B):
        max_j = None
        for j in range(DP):
            if S[i] <= sorted_max_seqlen_list[j]:
                max_j = j
            else:
                break
        assert max_j != None, f"can't find a strategy to run with seqlen {S[i]}"
        J.append(max_j)
        # print(f"seq {i} len is {batch_seqlen_array[i]}, and the max relative strategy id of it is {max_j}")
    
    # 以下代码deprecated
    # 由于线性规划求解开销过大因此采用上面的随机greedy算法
    # 定义最终优化的目标f
    def f(strategy_id, select_list, seqlen_list, max_seqlen=None):
        time_sum = 0
        for select, seqlen in zip(select_list, seqlen_list):
            time_sum += select * dynamic_strategy_time_cost(strategy_pool, strategy_id, seqlen) 
        if max_seqlen != None:
            time_sum += dynamic_strategy_time_cost(strategy_pool, strategy_id, max_seqlen) * (strategy_pool['strategies'][strategy_id]['pp'] - 1)
        return time_sum
    
    start_time = time.time()
    # 创建问题实例
    prob = pulp.LpProblem("ILP_Problem", pulp.LpMinimize)
    # 定义变量m_{i,j}
    m = pulp.LpVariable.dicts("m", (range(B), range(DP)), cat='Binary')
    # 定义辅助变量Z用于表示优化函数f的最大值
    Z = pulp.LpVariable("Z", lowBound=0)
    # 定义新的辅助变量Y_j表示当前j个数据并行组分到的最大的seq长度
    Y = pulp.LpVariable.dicts("Y", range(DP), lowBound=0)
    # 添加约束条件
    # sum(m_{i,j}) = 1 for all i
    for i in range(B):
        prob += pulp.lpSum(m[i][j] for j in range(J[i] + 1)) == 1
    # 添加约束条件
    # Y_j >= m_{i,j} * S_i for all i, j
    # 无法求解平方项的线性规划
    for j in range(DP):
        for i in range(B):
            prob += Y[j] >= m[i][j] * S[i]
    # 添加目标函数和相关约束
    # minimize Z and Z >= f(...)
    for j in range(DP):
        prob += Z >= f(sorted_strategy_id_list[j], [m[i][j] for i in range(B)], S, max_seqlen=Y[j])
        # prob += Z >= f(sorted_strategy_id_list[j], [m[i][j] for i in range(B)], S, max_seqlen=sorted_max_seqlen_list[j])
        # prob += Z >= f(sorted_strategy_id_list[j], [m[i][j] for i in range(B)], S)
    # 设置目标函数
    prob += Z
    # 求解问题
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=30))
    # prob.solve(pulp.PULP_CBC_CMD(msg=False))
    end_time = time.time()

    # 输出结果
    print(f"Status: {pulp.LpStatus[prob.status]}, time cost is {end_time - start_time}s")
    '''
    for i in range(B):
        for j in range(DP):
            print(f"m_{i}_{j} = {pulp.value(m[i][j])}")
    '''
    # print(f"Objective value: {pulp.value(prob.objective)}")
    for j in range(DP):
        pipeline_time_cost = f(sorted_strategy_id_list[j], [pulp.value(m[i][j]) for i in range(B)], S, max_seqlen=sorted_max_seqlen_list[j])
        # print(f"Estimiated value of {j}-th dp is: {pipeline_time_cost}")
    
    cur_column = sorted_cur_strategy_relative_id
    res = []
    for i in range(B):
        if pulp.value(m[i][cur_column]):
            res.append(i)
    assert len(res) > 0, "currently not support zero seqs"
    # res = list(range(B // DP * sorted_cur_strategy_relative_id, B // DP * (sorted_cur_strategy_relative_id + 1)))
    return (pulp.value(prob.objective), res)

def solve_v_micro_batches_scip(seqs: List[int], costs: List[float], max_seqlen: int, util_seqlen: int, v: int):
    u = len(seqs)
    model = Model("Minimize_Max_Cost")
    model.setParam('limits/time', 0.5)  # 设置求解时间限制为5秒
    model.setParam('display/verblevel', 0)  # 禁止显示求解过程中的消息
    o = {}
    for i in range(u):
        for j in range(v):
            o[i, j] = model.addVar(vtype="BINARY", name=f"o_{i}_{j}")
    max_cost = model.addVar(vtype="CONTINUOUS", name="max_cost", lb=0)
    for j in range(v):
        model.addCons(sum(o[i, j] * costs[i] for i in range(u)) <= max_cost)
    for i in range(u):
        model.addCons(sum(o[i, j] for j in range(v)) == 1)
    for j in range(v):
        # model.addCons(sum(o[i, j] * seqs[i] for i in range(u)) >= util_seqlen)
        model.addCons(sum(o[i, j] * seqs[i] for i in range(u)) <= max_seqlen)
    model.setObjective(max_cost, "minimize")
    model.optimize()
    # 检查求解状态
    status = model.getStatus()
    if status == "optimal":
        # 找到最优解
        max_cost_value = model.getObjVal()
        o_values = np.array([[model.getVal(o[i, j]) for j in range(v)] for i in range(u)])
        return max_cost_value, o_values
    elif status in ["timelimit", "gap limit reached", "node limit reached", "bestsollimit"]:
        # 时间限制或其他限制下找到一个可行解
        # print(f"Solver stopped with status: {status}. Returning the best feasible solution found.")
        if model.getNSols() > 0:  # 检查是否有可行解
            max_cost_value = model.getObjVal()
            o_values = np.array([[model.getVal(o[i, j]) for j in range(v)] for i in range(u)])
            return max_cost_value, o_values
        else:
            # print("No feasible solution found.")
            return float('inf'), None
    else:
        # 无法找到可行解或者问题无解
        # print(f"Problem status: {status}. No feasible solution found.")
        return float('inf'), None

def solve_v_micro_batches(seqs: List[int], costs: List[float], max_seqlen: int, util_seqlen: int, v: int):
    u = len(seqs)
    # Create the LP problem
    prob = pulp.LpProblem("Minimize_Max_Cost", pulp.LpMinimize)
    # Decision variables
    o = pulp.LpVariable.dicts("o", ((i, j) for i in range(u) for j in range(v)), cat='Binary')
    # Objective function: minimize the maximum cost of any micro batch
    max_cost = pulp.LpVariable("max_cost", lowBound=0)
    for j in range(v):
        prob += pulp.lpSum(o[i, j] * costs[i] for i in range(u)) <= max_cost
    prob += max_cost
    # Constraints
    for i in range(u):
        prob += pulp.lpSum(o[i, j] for j in range(v)) == 1
    for j in range(v):
        # prob += pulp.lpSum(o[i, j] * seqs[i] for i in range(u)) >= util_seqlen
        prob += pulp.lpSum(o[i, j] * seqs[i] for i in range(u)) <= max_seqlen
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=0.5))
    if pulp.LpStatus[prob.status] in ["Optimal", "Integer Feasible", "Stopped"]:
        max_cost_value = pulp.value(max_cost)
        if max_cost_value is not None:
            o_values = np.array([[pulp.value(o[i, j]) for j in range(v)] for i in range(u)])
            return max_cost_value, o_values
        else:
            return float('inf'), None 
    else:
        # print(f"cannot put {seqs} into {v} micro batches and reach the full utilization")
        return float('inf'), None

def batching_strategy(strategy_pool, strategy_id: int, seqs: List[int], max_seqlen: int):
    strategy = strategy_pool['strategies'][strategy_id]
    tp = strategy['tp']
    pp = strategy['pp']
    util_seqlen = strategy_pool['cluster_config']['utilization_seqlen'][f'tp{tp}']
    total_tokens = sum(seqs)
    min_num_micro_batches = max(1, (total_tokens + max_seqlen - 1) // max_seqlen)
    max_num_micro_batches = min(len(seqs), total_tokens // util_seqlen)
    max_num_micro_batches = max(min_num_micro_batches, max_num_micro_batches)
    num_micro_batches_enum = range(min_num_micro_batches, min(max_num_micro_batches + 1, min_num_micro_batches + 16))
    costs = [dynamic_strategy_time_cost(strategy_pool, strategy_id, seq) for seq in seqs]
    results = []
    start_time = time.time()
    if MAX_WORKERS == 1:
        results = []
        for v in num_micro_batches_enum:
            result = solve_v_micro_batches(seqs, costs, max_seqlen, util_seqlen, v)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=min(len(num_micro_batches_enum), MAX_WORKERS)) as executor:
            futures = [executor.submit(solve_v_micro_batches, seqs, costs, max_seqlen, util_seqlen, v) for v in num_micro_batches_enum]
            results = [future.result() for future in futures]
    end_time = time.time()
    print(f"micro-batching time cost is {end_time - start_time}s")
    optimal_e2e_cost = float('inf')
    optimal_o = None # optimal option (o_i_j means put i-th seq in j-th micro batch)
    optimal_v = None # optimal num micro batched
    for v, (max_cost_value, o_values) in zip(list(num_micro_batches_enum), results):
        if max_cost_value < float('inf'):
            e2e_cost = (max_cost_value + strategy['c']) * (pp - 1 + v)
            # print(f"form {v} micro batches will have e2e cost {e2e_cost}")
            if e2e_cost < optimal_e2e_cost:
                optimal_e2e_cost = e2e_cost
                optimal_o = o_values
                optimal_v = v
    if not isinstance(optimal_o, np.ndarray):
        # raise RuntimeError("cannot gurantee the sequence ultilization for all DP")
        # sys.exit(1)
        return float('inf'), None
    # print(f"Optimal e2e cost of the pipeline is {optimal_e2e_cost}")
    return optimal_e2e_cost, optimal_o

if __name__ == '__main__':
    # 读取并打印strategy数据
    file_path = 'strategy_pool_32b.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("Read strategy data:")
    print(json.dumps(data, indent=4))
    
    # Example usage
    seqs = [1000, 2000, 3000, 4000, 5000]
    max_seqlen = 6000
    strategy_id = 1

    optimal_o = batching_strategy(data, strategy_id, seqs, max_seqlen)
    print("Optimal Assignment Matrix (o):", optimal_o)

