
import argparse

import numpy as np
import torch


# 遍历数据集
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from data_utils.multimodal_dataset.NExTQADataset_without_cache import NExTQADataset
from data_utils import build_tokenizer
from data_utils import get_sorted_by_text_and_video, get_bucket



def plot_2d_list_as_bars(data_2d, name):
    """
    将二维列表按列绘制成柱状图，每行用不同颜色表示
    
    参数:
        data_2d: 二维列表
        title_prefix: 图表标题的前缀文字
    """
    import matplotlib.pyplot as plt
    import numpy as np


    plt.figure(figsize=(15, 8))
    
    print(name)
    print(data_2d)
    # 为每一行使用不同颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_2d)))
    
    # 根据行数动态调整柱子宽度和间距
    bar_width = max(0.8 / len(data_2d), 0.1)  # 确保柱子不会太窄
    column_spacing = max(bar_width * len(data_2d) * 1.5, 0.5)  # 根据柱子总宽度调整列间距
    
    # 遍历每一列
    for i in range(len(data_2d[0])):
        # 获取每行第i列的数据
        column_data = [row[i] for row in data_2d]
        
        # 计算当前列的基准x位置
        x_base = i * column_spacing
        
        # 在当前列内,每行的柱子交错排列
        bars = []  # 存储每行的柱状图对象用于添加图例
        for j in range(len(data_2d)):
            x = x_base + j * bar_width
            bar = plt.bar(x, column_data[j], bar_width, color=colors[j])
            bars.append(bar)
    
    # 添加图例
    plt.legend(bars, [f'dp{i}' for i in range(len(data_2d))])
    
    # 设置图表标题和标签
    plt.title(name)
    plt.xlabel('mini batch')
    plt.ylabel('sequence lengths')
    
    # 保存图表
    try:
        plt.savefig(f'/home/pkuhetu/njw1123/hetu_mm/examples/mllm/images/{name}.png')
        print(f"成功保存图表到 {name}.png")
    except Exception as e:
        print(f"保存图表失败: {str(e)}")
    finally:
        plt.close()

def process_sample(dataset, idx):
    try:
        data = dataset[idx]
        return data
    except Exception as e:
        print(f"处理索引 {idx} 时出错: {str(e)}")
        return None
    
def analyze_length(global_batch_size_, vision_max_seqlen_, text_max_seqlen_):

    tokenizer = None
    ROOT_PATH ="/home/pkuhetu/njw1123/hetu_mm/python/hetu/engine/data"
    args = argparse.Namespace()
    args.tokenizer_type = "GPT2BPETokenizer"
    args.vocab_file = ROOT_PATH + "/gpt2-vocab.json"
    args.merge_file = ROOT_PATH + "/gpt2-merges.txt"
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.rank = 0
    args.patch_size = 14
    args.temporal_patch_size = 2
    args.vision_max_seqlen = vision_max_seqlen_
    args.text_max_seqlen = text_max_seqlen_
    args.batching_method = 2


    tokenizer = build_tokenizer(args)
    dataset = NExTQADataset(tokenizer=tokenizer, args=args)

    assert global_batch_size_ < 128
    global_batch_num = 512 // global_batch_size_
    global_batch_size = global_batch_size_
    # 遍历每个global batch

    vision_len = []
    for batch_idx in range(global_batch_num):
        # 计算当前batch的索引范围
        start_idx = batch_idx * global_batch_size
        end_idx = min((batch_idx + 1) * global_batch_size, len(dataset))
        batch_indices = list(range(start_idx, end_idx))
        
        # 使用进程池并行处理当前batch的样本
        with Pool(16) as pool:
            batch_data = pool.map(partial(process_sample, dataset), batch_indices)
        
        # 过滤掉None值并添加到结果中
        # 过滤掉None值
        batch_data = [x for x in batch_data if x is not None]
        
        # 按key合并数据
        merged_batch = {}
        for key in batch_data[0].keys():
            merged_batch[key] = torch.Tensor(np.array([sample[key] for sample in batch_data]))
            print("key", merged_batch[key].shape)
        
        # print("merged data length", list(merged_batch["video_len"].numpy()))
        vision_len.extend(list(merged_batch["video_len"].numpy()))

    # 绘制视频长度分布的柱状图
    plt.figure(figsize=(15, 8))
    plt.hist(vision_len, bins=50, color='blue', alpha=0.7)
    plt.title(f'Distribution of Video Sequence Length (vidoe_max_seqlen={vision_max_seqlen_})')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    
    # Save the plot
    try:
        plt.savefig(f'/home/pkuhetu/njw1123/hetu_mm/examples/mllm/images/video_maxseq{vision_max_seqlen_}.png')
        print(f"Successfully saved video length distribution plot")
    except Exception as e:
        print(f"Failed to save plot: {str(e)}")
    finally:
        plt.close()



def analyze(global_batch_size_, vision_dp_size_, llm_dp_size_, vision_max_seqlen_, text_max_seqlen_):



    tokenizer = None
    alignment = 128
    IMAGE_TOKEN = -200
    ROOT_PATH ="/home/pkuhetu/njw1123/hetu_mm/python/hetu/engine/data"
    args = argparse.Namespace()
    args.tokenizer_type = "GPT2BPETokenizer"
    args.vocab_file = ROOT_PATH + "/gpt2-vocab.json"
    args.merge_file = ROOT_PATH + "/gpt2-merges.txt"
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.rank = 0
    args.patch_size = 14
    args.temporal_patch_size = 2
    args.vision_max_seqlen = vision_max_seqlen_
    args.text_max_seqlen = text_max_seqlen_
    args.batching_method = 2


    tokenizer = build_tokenizer(args)
    dataset = NExTQADataset(tokenizer=tokenizer, args=args)

    assert global_batch_size_ < 128
    global_batch_num = 128 // global_batch_size_
    global_batch_size = global_batch_size_
    vision_dp_size = vision_dp_size_
    llm_dp_size = llm_dp_size_

    # 遍历每个global batch

    vision_len = []
    for i in range(vision_dp_size):
        vision_len.append([])
    llm_len = []
    for i in range(llm_dp_size):
        llm_len.append([])

    for batch_idx in range(global_batch_num):
        # 计算当前batch的索引范围
        start_idx = batch_idx * global_batch_size
        end_idx = min((batch_idx + 1) * global_batch_size, len(dataset))
        batch_indices = list(range(start_idx, end_idx))
        
        # 使用进程池并行处理当前batch的样本
        with Pool(16) as pool:
            batch_data = pool.map(partial(process_sample, dataset), batch_indices)
        
        # 过滤掉None值并添加到结果中
        # 过滤掉None值
        batch_data = [x for x in batch_data if x is not None]
        
        # 按key合并数据
        merged_batch = {}
        for key in batch_data[0].keys():
            merged_batch[key] = torch.Tensor(np.array([sample[key] for sample in batch_data]))
            print("key", merged_batch[key].shape)
        

        sorted_batch = get_sorted_by_text_and_video(merged_batch, tokenizer.pad)
        bucket = get_bucket(sorted_batch, tokenizer.pad, args.vision_max_seqlen, args.text_max_seqlen, IMAGE_TOKEN, alignment)
        bucket.pack_data(args.batching_method, vision_dp_size, llm_dp_size)
        vision_cu_seqlens_list = bucket.get_packed_vision_cu_seqlens_list()
        llm_cu_seqlens_list = bucket.get_packed_llm_cu_seqlens_list()
        assert len(vision_cu_seqlens_list) % vision_dp_size == 0
        assert len(llm_cu_seqlens_list) % llm_dp_size == 0
        llm_per_dp = len(llm_cu_seqlens_list) // llm_dp_size
        vision_per_dp = len(vision_cu_seqlens_list) // vision_dp_size
        assert llm_per_dp == vision_per_dp
        
        for i in range(len(llm_cu_seqlens_list)):
            llm_len[i % llm_dp_size].append(llm_cu_seqlens_list[i][-1])
        for i in range(len(vision_cu_seqlens_list)):
            vision_len[i % vision_dp_size].append(vision_cu_seqlens_list[i][-1])

    prefix_name = f"global_batch_size_{global_batch_size}_llm_dp_size_{llm_dp_size}_vision_dp_size_{vision_dp_size}_llm_max_seq_{args.text_max_seqlen}_vision_max_seq_{args.vision_max_seqlen}"
    plot_2d_list_as_bars(llm_len, prefix_name + "_llm")
    plot_2d_list_as_bars(vision_len, prefix_name + "_vision")

gbs = [32, 64, 128]
vision_dp_size = [1, 2, 3, 4]
llm_dp_size = [1, 2, 3, 4]
# max_seq = [[4 * 1024, 5 * 1024], [4 * 1024, 8 * 1024], [4 * 1024, 16 * 1024], [8 * 1024, 9 * 1024], [8 * 1024, 16 * 1024], [8 * 1024, 32 * 1024], [16 * 1024, 17 * 1024], [16 * 1024, 32 * 1024], [16 * 1024, 64 * 1024], [32 * 1024, 33 * 1024], [32 * 1024, 64 * 1024], [32 * 1024, 128 * 1024], [64 * 1024, 65 * 1024], [64 * 1024, 128 * 1024]]
max_seq = [[64 * 1024, 65 * 1024]]

# for gb in gbs:
#     for vdp in vision_dp_size:
#         for ldp in llm_dp_size:
#             for seq in max_seq:
#                 try:
#                     analyze(gb, vdp, ldp, seq[0], seq[1])
#                 except Exception as e:
#                     print(f"分析失败: {str(e)}")

# analyze_length(64, 4 * 1024, 5 * 1024)
analyze_length(64, 8 * 1024, 9 * 1024)
analyze_length(64, 16 * 1024, 17 * 1024)
analyze_length(64, 32 * 1024, 33 * 1024)
analyze_length(64, 64 * 1024, 65 * 1024)