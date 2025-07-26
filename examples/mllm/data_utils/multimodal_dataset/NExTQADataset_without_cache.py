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
    def __init__(self, tokenizer, text_path = "/home/pkuhetu/njw1123/hetu_mm/python/hetu/engine/data/multimodal_data/NExTQA/MC/test-00000-of-00001.parquet",
        vision_path = "/home/pkuhetu/njw1123/hetu_mm/python/hetu/engine/data/multimodal_data/NExTQA/NExTVideo", args = None):
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
        self.vision_max_seq_len = args.vision_max_seqlen
        self.text_max_seq_len = args.text_max_seqlen
        
        # 读取parquet文件
        self.df = pd.read_parquet(text_path)
        self.filter_invalid_videos()

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

    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            dict: 包含文本数据的字典
        """
        
        # 读取视频数据
        item = self.df.iloc[idx]
        video_item_path = os.path.join(self.vision_path, str(item['video']) + ".mp4")
        frames = self.open_video(video_item_path)
        frames = np.transpose(frames, (0, 3, 1, 2))
        
        # 处理文本和视频
        texts = "question: " + item['question'] + " answer: " + str(item['answer'])
        text_inputs_data, image_inputs_data, videos_inputs_data = self.processor(images=None, texts=texts, videos=[frames])

        # 处理视频数据
        video_length = videos_inputs_data['video_grid_thws'][0][0] * videos_inputs_data['video_grid_thws'][0][1] * videos_inputs_data['video_grid_thws'][0][2]
        video_hidden_size = videos_inputs_data['video_pixel_values'][0].shape[1]

        vision_tokens = None
        if video_length <= self.vision_max_seq_len:
            pad_rows = np.ones((self.vision_max_seq_len - video_length, video_hidden_size), dtype=np.float32) * self.tokenizer.pad
            vision_tokens = np.concatenate([videos_inputs_data['video_pixel_values'][0].astype(np.float32), pad_rows], axis=0)
        else:
            vision_tokens = videos_inputs_data['video_pixel_values'][0][:self.vision_max_seq_len].astype(np.float32)
            video_length = self.vision_max_seq_len

        # 处理文本数据
        text_tokens = np.array(text_inputs_data[0], dtype=np.float32)
        text_length = text_tokens.shape[0]  
        if text_length <= self.text_max_seq_len:
            pad_rows = np.ones((self.text_max_seq_len - text_length), dtype=np.float32) * self.tokenizer.pad
            text_tokens = np.concatenate([text_tokens, pad_rows], axis=0)
        else:
            text_tokens = text_tokens[:self.text_max_seq_len]
            text_length = self.text_max_seq_len


        return {
            'text': text_tokens,
            'text_label': text_tokens,
            'text_len': text_length,
            'video': vision_tokens,
            'video_len': video_length
        }