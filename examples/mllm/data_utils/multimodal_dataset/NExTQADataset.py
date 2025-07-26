import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision
import av
import pickle
import time
from data_utils import HetuMLLMProcessor, HetuImageProcessor, build_tokenizer

class NExTQADataset(Dataset):
    def __init__(self, tokenizer, text_path = "/home/gehao/njw1123/merge_all/python/hetu/data/multimodal_data/NExTQA/MC/test-00000-of-00001.parquet",
        vision_path = "/home/gehao/njw1123/merge_all/python/hetu/data/multimodal_data/NExTQA/NExTVideo", args = None):
        """
        初始化NExTQA数据集
        Args:
            data_path: parquet文件路径
        """
        super().__init__()
        self.text_path = text_path
        self.vision_path = vision_path
        self.tokenizer = tokenizer
        self.patch_size = args.patch_size
        self.temporal_patch_size = args.temporal_patch_size
        self.processor = HetuMLLMProcessor(image_processor=HetuImageProcessor(patch_size=self.patch_size, temporal_patch_size=self.temporal_patch_size), tokenizer=tokenizer, chat_template=None)
        self.vision_max_seq_len = args.vision_max_seqlen // 2
        self.text_max_seq_len = args.text_max_seqlen // 2
        
        
        # 读取parquet文件
        self.df = pd.read_parquet(text_path)
        self.filter_invalid_videos()
        
        # 生成cache文件路径
        cache_dir = os.path.dirname(vision_path)
        cache_name_prefix = f"nextqa_cache_vlen{self.vision_max_seq_len}_tlen{self.text_max_seq_len}_{self.patch_size}_{self.temporal_patch_size}"
        self.text_cache_path = os.path.join(cache_dir, cache_name_prefix + "_text.npy")
        self.text_label_cache_path = os.path.join(cache_dir, cache_name_prefix + "_text_label.npy") 
        self.text_len_cache_path = os.path.join(cache_dir, cache_name_prefix + "_text_len.npy")
        self.video_cache_path = os.path.join(cache_dir, cache_name_prefix + "_video.npy")
        self.video_len_cache_path = os.path.join(cache_dir, cache_name_prefix + "_video_len.npy")
        
        # 如果cache文件存在则直接加载,否则生成cache
        if os.path.exists(self.text_cache_path):
            print(f"Loading cache from {cache_name_prefix}")
            # 分别加载各个数组
            # 使用mmap_mode='r'允许多进程共享读取内存映射文件
            try:
                self.text_data = np.load(self.text_cache_path, mmap_mode='r')
                self.text_label_data = np.load(self.text_label_cache_path, mmap_mode='r') 
                self.text_len_data = np.load(self.text_len_cache_path, mmap_mode='r')
                self.video_data = np.load(self.video_cache_path, mmap_mode='r')
                self.video_len_data = np.load(self.video_len_cache_path, mmap_mode='r')
            except Exception as e:
                # 如果加载失败,等待1秒后重试,最多重试3次
                for _ in range(3):
                    time.sleep(1)
                    try:
                        self.text_data = np.load(self.text_cache_path, mmap_mode='r')
                        self.text_label_data = np.load(self.text_label_cache_path, mmap_mode='r')
                        self.text_len_data = np.load(self.text_len_cache_path, mmap_mode='r') 
                        self.video_data = np.load(self.video_cache_path, mmap_mode='r')
                        self.video_len_data = np.load(self.video_len_cache_path, mmap_mode='r')
                        break
                    except Exception:
                        continue
                else:
                    raise e
        else:
            print(f"Generating cache to {cache_name_prefix}")
            if args.rank == 0:
                self.generate_cache()
                print(f"Saving cache to {cache_name_prefix}")
                np.save(self.text_cache_path, self.text_data)
                np.save(self.text_label_cache_path, self.text_label_data)
                np.save(self.text_len_cache_path, self.text_len_data)
                np.save(self.video_cache_path, self.video_data)
                np.save(self.video_len_cache_path, self.video_len_data)
            else:
                # 等待所有cache文件都存在且完整写入
                cache_files = [self.text_cache_path, self.text_label_cache_path, 
                             self.text_len_cache_path, self.video_cache_path,
                             self.video_len_cache_path]
                while True:
                    all_files_ready = True
                    for f in cache_files:
                        if not os.path.exists(f):
                            all_files_ready = False
                            break
                        # 检查文件是否完整写入
                        try:
                            np.load(f, mmap_mode='r')
                        except Exception:
                            all_files_ready = False
                            break
                    if all_files_ready:
                        break
                    time.sleep(1)
                
                # 加载cache文件
                self.text_data = np.load(self.text_cache_path, mmap_mode='r')
                self.text_label_data = np.load(self.text_label_cache_path, mmap_mode='r')
                self.text_len_data = np.load(self.text_len_cache_path, mmap_mode='r')
                self.video_data = np.load(self.video_cache_path, mmap_mode='r')
                self.video_len_data = np.load(self.video_len_cache_path, mmap_mode='r')

    def filter_invalid_videos(self):
        """
        过滤掉不存在对应视频文件的数据
        """
        valid_indices = []
        for idx in range(len(self.df)):
            video_path = os.path.join(self.vision_path, str(self.df.iloc[idx]['video']) + ".mp4")
            if os.path.exists(video_path):
                valid_indices.append(idx)
                
        # 只保留有效的行
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def open_video(self, video_item_path):
        container = av.open(video_item_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
        frames = np.stack(frames)
        meta = {'video_fps': float(container.streams.video[0].average_rate)}
        container.close()
        # 每秒采样2帧
        # sample_rate = int(meta['video_fps'] / 2)
        sample_rate = int(meta['video_fps'] * 4)  # 将浮点数转换为整数
        if frames.shape[0] > sample_rate:
            frames = frames[::sample_rate]
        else:
            frames = frames[0]
        return frames

    def generate_cache(self):
        """
        生成所有数据的cache,使用多线程加速处理
        """
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def process_single_item(idx):
            item = self.df.iloc[idx]
            video_item_path = os.path.join(self.vision_path, str(item['video']) + ".mp4")
            
            frames = self.open_video(video_item_path)
            frames = np.transpose(frames, (0, 3, 1, 2))
            texts = "question: " + item['question'] + " answer: " + str(item['answer'])
            text_inputs_data, image_inputs_data, videos_inputs_data = self.processor(images=None, texts=texts, videos=[frames])

            video_length = videos_inputs_data['video_grid_thws'][0][0] * videos_inputs_data['video_grid_thws'][0][1] * videos_inputs_data['video_grid_thws'][0][2]
            video_hidden_size = videos_inputs_data['video_pixel_values'][0].shape[1]

            vision_tokens = None
            # 将video length向下取整到128的倍数
            if video_length <= self.vision_max_seq_len:
                # 先截断到128的倍数
                padded_video_length = (video_length // 128) * 128
                video_pixels = videos_inputs_data['video_pixel_values'][0][:padded_video_length]
                # 然后填充到vision_max_seq_len
                pad_rows = np.ones((self.vision_max_seq_len - padded_video_length, video_hidden_size)) * self.tokenizer.pad
                vision_tokens = np.concatenate([video_pixels, pad_rows], axis=0)
                video_length = padded_video_length
            else:
                vision_tokens = videos_inputs_data['video_pixel_values'][0][:self.vision_max_seq_len]
                video_length = self.vision_max_seq_len

            text_tokens = np.array(text_inputs_data[0])
            text_length = text_tokens.shape[0]  
            if text_length <= self.text_max_seq_len:
                pad_rows = np.ones((self.text_max_seq_len - text_length)) * self.tokenizer.pad
                text_tokens = np.concatenate([text_tokens, pad_rows], axis=0)
            else:
                text_tokens = text_tokens[:self.text_max_seq_len]
                text_length = self.text_max_seq_len

            print(f"Cached {idx}/{len(self.df)}: text_length: {text_length}, video_length: {video_length}")
            return text_tokens, text_tokens, text_length, vision_tokens, video_length

        # 使用线程池并行处理
        text_data = []
        text_label_data = []
        text_len_data = []
        video_data = []
        video_len_data = []
        
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(process_single_item, idx) for idx in range(100)]
            for future in futures:
                text, text_label, text_len, video, video_len = future.result()
                text_data.append(text)
                text_label_data.append(text_label)
                text_len_data.append(text_len)
                video_data.append(video)
                video_len_data.append(video_len)
                
        self.text_data = np.array(text_data, dtype=np.float32)
        self.text_label_data = np.array(text_label_data, dtype=np.float32)
        self.text_len_data = np.array(text_len_data)
        self.video_data = np.array(video_data, dtype=np.float32)
        self.video_len_data = np.array(video_len_data)
    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            dict: 包含文本数据的字典
        """
        text = self.text_data[idx]
        text_label = self.text_label_data[idx]
        text_len = self.text_len_data[idx] 
        video = self.video_data[idx]
        video_len = self.video_len_data[idx]
        

        return {
            'text': text,
            'text_label': text_label,
            'text_len': text_len,
            'video': video,
            'video_len': video_len
        }