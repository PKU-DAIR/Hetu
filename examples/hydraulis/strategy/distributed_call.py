import os
import pickle
import time
import fcntl
import hetu as ht
from typing import Callable, Any, Tuple, Dict

func_call_file = "./func_call"
write_tag = 0
read_tag = 0

# workaround
# hope to leverage grpc in the future
def distributed_call(distributed_status: Tuple[int, int, Dict[int, int]], func: Callable, *args: Any, **kwargs: Any):
    # synchronize all processes
    # but seems doesn't work
    ht.global_comm_barrier_rpc() 
    global write_tag, read_tag
    start_time = time.time()
    gpu_id, dp_id, dp_representive_gpu = distributed_status
    path = func_call_file + f"_{func.__name__}_dp{dp_id}.pkl"
    if gpu_id == dp_representive_gpu[dp_id]:
        # representive rank process call the function and write the result to the file
        result = func(*args, **kwargs)
        with open(path, 'wb') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                pickle.dump((result, write_tag), file)
                file.flush()
                os.fsync(file.fileno())
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)
        write_tag += 1
    ht.global_comm_barrier_rpc() 
    if gpu_id != dp_representive_gpu[dp_id]:
        while True:
            try:
                with open(path, 'rb') as file:
                    try:
                        fcntl.flock(file, fcntl.LOCK_SH)
                        result, tag = pickle.load(file)
                    finally:
                        fcntl.flock(file, fcntl.LOCK_UN)
            except Exception as e:
                # print("Exception raise")
                time.sleep(1)  # 等待文件写入完成
                continue
            if tag == read_tag:
                break
            else:
                # print(f"read tag = {read_tag} but file tag = {tag}")
                time.sleep(1)  # 等待文件写入完成
        read_tag += 1
    ht.global_comm_barrier_rpc() 
    end_time = time.time()
    print(f"Distributed func call {func.__name__} time cost: {end_time - start_time}s")
    return result