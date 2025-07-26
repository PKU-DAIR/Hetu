import numpy as np
from typing import List

class MultimodalData:
    def __init__(self, vision: np.ndarray, vision_len: int, text: np.ndarray, text_label: np.ndarray, text_len: int, mask_token: int, pad_token: int):
        self.vision = vision
        self.vision_len = vision_len
        self.text = text
        self.text_label = text_label
        self.text_len = text_len
        self.mask_token = mask_token
        self.pad_token = pad_token
        assert self.vision.shape[0] == self.vision_len
        assert self.text.shape[0] == self.text_len
        assert self.text_label.shape[0] == self.text_len


    def get_vision_input(self):
        return self.vision

    def get_llm_input_and_label(self):
        input = np.concatenate([np.array([self.mask_token] * self.vision_len), self.text])
        label = np.concatenate([np.array([self.mask_token] * self.vision_len), self.text_label]) 
        return input, label

    def text_and_video_len(self):
        return self.vision_len + self.text_len
    
    def text_len(self):
        return self.text_len
    
    def vision_len(self):
        return self.vision_len
    
    def pad_vision(self, max_seqlen: int):
        pad_seqlen = max_seqlen - self.vision_len
        if pad_seqlen > 0:
            # vision是两个维度的,需要在第一个维度上pad
            hidden_size = self.vision.shape[1]
            pad_rows = np.ones((pad_seqlen, hidden_size)) * self.pad_token
            self.vision = np.concatenate([self.vision, pad_rows], axis=0)
            self.vision_len = max_seqlen
    
    def pad_text(self, max_seqlen: int):
        pad_seqlen = max_seqlen - self.text_len
        if pad_seqlen > 0:
            self.text = np.concatenate([self.text, np.array([self.pad_token] * pad_seqlen)])
            self.text_len = max_seqlen
    
    def get_llm_input_mask(self):
        return np.concatenate([np.array([1] * self.vision_len), np.array([0] * self.text_len)])
    



class MultimodalBucket:
    def __init__(self, pad_token: int, vision_max_seqlen: int, text_max_seqlen: int, alignment: int = 128):
        self._pad_token = pad_token
        self._vision_max_seqlen = vision_max_seqlen
        self._text_max_seqlen = text_max_seqlen
        self._alignment = alignment
        self.batch = []
        self._padded_vision_batch = None
        self._padded_llm_batch = None
        self._padded_vision_cu_seqlens_list = None
        self._padded_llm_cu_seqlens_list = None
        self.llm_input_mask = None
        self._packed_vision_batch = None
        self._packed_llm_batch = None
        self._packed_llm_label_batch = None
        self._packed_vision_cu_seqlens_list = None
        self._packed_llm_cu_seqlens_list = None
        self._vision_packing_slice_list = None
        self._llm_packing_slice_list = None
        
    def add_data(self, vision_batch: np.ndarray, vision_batch_len: int, text_batch: np.ndarray, text_label: np.ndarray, text_batch_len: int, mask_token: int):
        self.batch.append(MultimodalData(vision_batch[0:vision_batch_len], vision_batch_len, text_batch[0:text_batch_len], text_label[0:text_batch_len], text_batch_len, mask_token, self._pad_token))
    
    def pad_data(self):
        padded_vision_batch = []
        padded_llm_batch = []
        padded_vision_cu_seqlens_list = []
        padded_llm_cu_seqlens_list = []
        llm_input_mask = []
        for data in self.batch:
            data.pad_vision(self._vision_max_seqlen)
            data.pad_text(self._text_max_seqlen)
            padded_vision_batch.append(data.vision)
            padded_llm_batch.append(data.text)
            padded_vision_cu_seqlens_list.append(data.vision_len)
            padded_llm_cu_seqlens_list.append(data.text_and_video_len())
            llm_input_mask.append(data.get_llm_input_mask())
        self._padded_vision_batch = padded_vision_batch
        self._padded_llm_batch = padded_llm_batch
        self._padded_vision_cu_seqlens_list = padded_vision_cu_seqlens_list
        self._padded_llm_cu_seqlens_list = padded_llm_cu_seqlens_list
        self.llm_input_mask = llm_input_mask
    
    def get_padded_vision_batch(self):
        assert self._padded_vision_batch != None, "please ensure you have padded the bucket"
        return self._padded_vision_batch
    
    def get_padded_llm_batch(self):
        assert self._padded_llm_batch != None, "please ensure you have padded the bucket"
        return self._padded_llm_batch
    
    def get_padded_vision_cu_seqlens_list(self):
        assert self._padded_vision_cu_seqlens_list != None, "please ensure you have padded the bucket"
        return self._padded_vision_cu_seqlens_list
    
    def get_padded_llm_cu_seqlens_list(self):
        assert self._padded_llm_cu_seqlens_list != None, "please ensure you have padded the bucket"
        return self._padded_llm_cu_seqlens_list

    def get_llm_input_mask(self):
        assert self.llm_input_mask != None, "please ensure you have padded the bucket"
        return self.llm_input_mask

    def packing_method_1(self):

        assert self._packed_vision_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_label_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_vision_cu_seqlens_list == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_cu_seqlens_list == None, "please ensure you have not packed the bucket"
        assert self.llm_input_mask == None, "please ensure you have not packed the bucket"  

        packed_vision_input_batch = []
        packed_llm_input_batch = []
        packed_llm_label_batch = []
        packed_vision_cu_seqlens_list = []
        packed_llm_cu_seqlens_list = []
        packed_llm_input_mask = []
        
        is_visited = set()
        for i in range(len(self.batch)):
            if i in is_visited:
                continue

            vision_packed_seqs = [self.batch[i].get_vision_input()]
            llm_packed_input_seqs = [self.batch[i].get_llm_input_and_label()[0]]
            llm_packed_label_seqs = [self.batch[i].get_llm_input_and_label()[1]]
            vision_cu_seqlens = [0, self.batch[i].vision_len]
            llm_cu_seqlens = [0, self.batch[i].text_and_video_len()]
            llm_input_mask = [self.batch[i].get_llm_input_mask()]
            is_visited.add(i)
            for j in reversed(range(i + 1, len(self.batch))):
                if j not in is_visited and vision_cu_seqlens[-1] + self.batch[j].vision_len <= self._vision_max_seqlen and llm_cu_seqlens[-1] + self.batch[j].text_and_video_len() <= self._text_max_seqlen:
                    vision_packed_seqs.append(self.batch[j].get_vision_input())
                    vision_cu_seqlens.append(vision_cu_seqlens[-1] + self.batch[j].vision_len)
                    llm_packed_input_seqs.append(self.batch[j].get_llm_input_and_label()[0])
                    llm_packed_label_seqs.append(self.batch[j].get_llm_input_and_label()[1])
                    llm_cu_seqlens.append(llm_cu_seqlens[-1] + self.batch[j].text_and_video_len())
                    llm_input_mask.append(self.batch[j].get_llm_input_mask())
                    is_visited.add(j)

            if vision_cu_seqlens[-1] % self._alignment != 0:
                pad_seqlen = self._alignment - (vision_cu_seqlens[-1] % self._alignment)
                hidden_size = vision_packed_seqs[0].shape[1]
                pad_rows = np.ones((pad_seqlen, hidden_size)) * self._pad_token
                vision_packed_seqs[-1] = np.concatenate([vision_packed_seqs[-1], pad_rows], axis=0)
                vision_cu_seqlens[-1] = vision_cu_seqlens[-1] + pad_seqlen
                
            if llm_cu_seqlens[-1] % self._alignment != 0:
                pad_seqlen = self._alignment - (llm_cu_seqlens[-1] % self._alignment)
                llm_packed_input_seqs[-1] = np.concatenate([llm_packed_input_seqs[-1], np.array([self._pad_token] * pad_seqlen)])
                llm_packed_label_seqs[-1] = np.concatenate([llm_packed_label_seqs[-1], np.array([self._pad_token] * pad_seqlen)])
                llm_cu_seqlens[-1] = llm_cu_seqlens[-1] + pad_seqlen
                llm_input_mask[-1] = np.concatenate([llm_input_mask[-1], np.array([0] * pad_seqlen)])
            
            packed_vision_input_batch.append(np.concatenate(vision_packed_seqs))
            packed_llm_input_batch.append(np.concatenate(llm_packed_input_seqs))
            packed_llm_label_batch.append(np.concatenate(llm_packed_label_seqs))
            packed_vision_cu_seqlens_list.append(np.array(vision_cu_seqlens, dtype=np.int32))
            packed_llm_cu_seqlens_list.append(np.array(llm_cu_seqlens, dtype=np.int32))
            packed_llm_input_mask.append(np.concatenate(llm_input_mask))

        self._packed_vision_batch = packed_vision_input_batch
        self._packed_llm_batch = packed_llm_input_batch
        self._packed_llm_label_batch = packed_llm_label_batch
        self._packed_vision_cu_seqlens_list = packed_vision_cu_seqlens_list
        self._packed_llm_cu_seqlens_list = packed_llm_cu_seqlens_list
        self.llm_input_mask = packed_llm_input_mask    

    def check_vsion_packing(self):
        pass

    def packing_method_2(self, vision_dp_size = 1, llm_dp_size = 1):


        def find_postion(data_idx : int, cu_seqlens : list, is_vision : bool):
            
            for dp_id, cu_seqlen in enumerate(cu_seqlens):
                if is_vision:
                    if len(cu_seqlen) > 0 and cu_seqlen[-1] + self.batch[data_idx].vision_len <= self._vision_max_seqlen or len(cu_seqlen) == 0:
                        return dp_id
                else:
                    if len(cu_seqlen) > 0 and cu_seqlen[-1] + self.batch[data_idx].text_and_video_len() <= self._text_max_seqlen or len(cu_seqlen) == 0:
                        return dp_id
            return -1



        assert self._packed_vision_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_vision_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_label_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_vision_cu_seqlens_list == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_cu_seqlens_list == None, "please ensure you have not packed the bucket"
        assert self.llm_input_mask == None, "please ensure you have not packed the bucket"  

        packed_vision_input_batch = []
        packed_llm_input_batch = []
        packed_llm_label_batch = []
        packed_vision_cu_seqlens_list = []
        packed_llm_cu_seqlens_list = []
        packed_llm_input_mask = []
        
        is_visited = set()
        for i in range(len(self.batch)):
            packed_vision_input_mini_batch = []
            packed_llm_input_mini_batch = []
            packed_llm_label_mini_batch = []
            packed_vision_cu_seqlens_mini_batch = []
            packed_llm_cu_seqlens_mini_batch = []
            packed_llm_input_mask_mini_batch = []
            for j in range(vision_dp_size):
                packed_vision_input_mini_batch.append([])
                packed_vision_cu_seqlens_mini_batch.append([])
            for j in range(llm_dp_size):
                packed_llm_input_mini_batch.append([])
                packed_llm_label_mini_batch.append([])
                packed_llm_cu_seqlens_mini_batch.append([])
                packed_llm_input_mask_mini_batch.append([])

            find_data = False
            for j in range(i, len(self.batch)):
                if j in is_visited:
                    continue
                
                vision_dp_id = find_postion(j, packed_vision_cu_seqlens_mini_batch, True)
                llm_dp_id = find_postion(j, packed_llm_cu_seqlens_mini_batch, False)
                if vision_dp_id == -1 or llm_dp_id == -1:
                    continue
                find_data = True
                is_visited.add(j)
                packed_vision_input_mini_batch[vision_dp_id].append(self.batch[j].get_vision_input())
                if len(packed_vision_cu_seqlens_mini_batch[vision_dp_id]) == 0:
                    packed_vision_cu_seqlens_mini_batch[vision_dp_id].append(0)
                    packed_vision_cu_seqlens_mini_batch[vision_dp_id].append(self.batch[j].vision_len)
                else:
                    packed_vision_cu_seqlens_mini_batch[vision_dp_id].append(packed_vision_cu_seqlens_mini_batch[vision_dp_id][-1] + self.batch[j].vision_len)
                packed_llm_input_mini_batch[llm_dp_id].append(self.batch[j].get_llm_input_and_label()[0])
                packed_llm_label_mini_batch[llm_dp_id].append(self.batch[j].get_llm_input_and_label()[1])
                if len(packed_llm_cu_seqlens_mini_batch[llm_dp_id]) == 0:
                    packed_llm_cu_seqlens_mini_batch[llm_dp_id].append(0)
                    packed_llm_cu_seqlens_mini_batch[llm_dp_id].append(self.batch[j].text_and_video_len())
                else:
                    packed_llm_cu_seqlens_mini_batch[llm_dp_id].append(packed_llm_cu_seqlens_mini_batch[llm_dp_id][-1] + self.batch[j].text_and_video_len())
                packed_llm_input_mask_mini_batch[llm_dp_id].append(self.batch[j].get_llm_input_mask())

            if find_data:
                for j in range(vision_dp_size):
                    if len(packed_vision_input_mini_batch[j]) == 0:
                        vision_hidden_size = self.batch[0].get_vision_input().shape[1]
                        packed_vision_input_mini_batch[j].append(np.zeros((128, vision_hidden_size), dtype=np.float32))
                        packed_vision_cu_seqlens_mini_batch[j] = [0, 128]
                    else:
                        if packed_vision_cu_seqlens_mini_batch[j][-1] % self._alignment != 0:
                            pad_seqlen = self._alignment - (packed_vision_cu_seqlens_mini_batch[j][-1] % self._alignment)
                            hidden_size = packed_vision_input_mini_batch[j][0].shape[1]
                            pad_rows = np.ones((pad_seqlen, hidden_size)) * self._pad_token
                            packed_vision_input_mini_batch[j][-1] = np.concatenate([packed_vision_input_mini_batch[j][-1], pad_rows], axis=0)
                            packed_vision_cu_seqlens_mini_batch[j][-1] = packed_vision_cu_seqlens_mini_batch[j][-1] + pad_seqlen      
                                

                    packed_vision_input_batch.append(np.concatenate(packed_vision_input_mini_batch[j]))
                    packed_vision_cu_seqlens_list.append(np.array(packed_vision_cu_seqlens_mini_batch[j], dtype=np.int32))


                for j in range(llm_dp_size):
                    if len(packed_llm_input_mini_batch[j]) == 0:
                        packed_llm_input_mini_batch[j].append(np.ones((128), dtype=np.int32) * self._pad_token)
                        packed_llm_label_mini_batch[j].append(np.ones((128), dtype=np.int32) * self._pad_token)
                        packed_llm_cu_seqlens_mini_batch[j] = [0, 128]
                        packed_llm_input_mask_mini_batch[j].append(np.zeros((128), dtype=np.int32))
                    else:
                        if packed_llm_cu_seqlens_mini_batch[j][-1] % self._alignment != 0:
                            pad_seqlen = self._alignment - (packed_llm_cu_seqlens_mini_batch[j][-1] % self._alignment)
                            packed_llm_input_mini_batch[j][-1] = np.concatenate([packed_llm_input_mini_batch[j][-1], np.array([self._pad_token] * pad_seqlen)])
                            packed_llm_label_mini_batch[j][-1] = np.concatenate([packed_llm_label_mini_batch[j][-1], np.array([self._pad_token] * pad_seqlen)])
                            packed_llm_cu_seqlens_mini_batch[j][-1] = packed_llm_cu_seqlens_mini_batch[j][-1] + pad_seqlen
                            packed_llm_input_mask_mini_batch[j][-1] = np.concatenate([packed_llm_input_mask_mini_batch[j][-1], np.array([0] * pad_seqlen)])

                    packed_llm_input_batch.append(np.concatenate(packed_llm_input_mini_batch[j]))
                    packed_llm_label_batch.append(np.concatenate(packed_llm_label_mini_batch[j]))
                    packed_llm_cu_seqlens_list.append(np.array(packed_llm_cu_seqlens_mini_batch[j], dtype=np.int32))
                    packed_llm_input_mask.append(np.concatenate(packed_llm_input_mask_mini_batch[j]))

        for i in range(len(self.batch)):
            if i not in is_visited:
                raise ValueError(f"data {i} is not visited")
        

        self._packed_vision_batch = packed_vision_input_batch
        self._packed_llm_batch = packed_llm_input_batch
        self._packed_llm_label_batch = packed_llm_label_batch
        self._packed_vision_cu_seqlens_list = packed_vision_cu_seqlens_list
        self._packed_llm_cu_seqlens_list = packed_llm_cu_seqlens_list
        self.llm_input_mask = packed_llm_input_mask         


    def packing_method_3(self, vision_dp_size = 1, llm_dp_size = 1):


        def find_postion(data_idx : int, cu_seqlens : list, is_vision : bool):
            
            max_space = -1
            best_dp_id = -1
            for dp_id, seqlen in enumerate(cu_seqlens):
                if is_vision:
                    remaining = self._vision_max_seqlen - seqlen
                    if remaining >= self.batch[data_idx].vision_len and remaining > max_space:
                        max_space = remaining
                        best_dp_id = dp_id
                else:
                    remaining = self._text_max_seqlen - seqlen
                    if remaining >= self.batch[data_idx].text_and_video_len() and remaining > max_space:
                        max_space = remaining
                        best_dp_id = dp_id
            return best_dp_id

        def sorted_lista_by_listb(lista, listb):
            assert len(lista) == len(listb), "lista and listb must have the same length"
            
            # 对于二维列表，对每一行按照listb的顺序排序
            result = []
            for i in range(len(lista)):
                row = lista[i]
                indices = list(range(len(row)))
                sorted_indices = sorted(indices, key=lambda j: listb[i][j])
                result.append([row[j] for j in sorted_indices])
            
            return result

        def get_packed_cu_seqlens_mini_batch(packed_input_mini_batch, dp_size):
            # 初始化cu_seqlens列表，用于记录每个batch中序列的累积长度
            packed_cu_seqlens_mini_batch = [[] for _ in range(dp_size)]
            # 计算累积序列长度
            for j in range(dp_size):
                if len(packed_input_mini_batch[j]) > 0:
                    # 第一个元素从0开始
                    packed_cu_seqlens_mini_batch[j].append(0)
                    # 累积每个序列的长度
                    cur_len = 0
                    for input in packed_input_mini_batch[j]:
                        cur_len += input.shape[0]
                        packed_cu_seqlens_mini_batch[j].append(cur_len)
                else:
                    packed_cu_seqlens_mini_batch[j].append(0)
            return packed_cu_seqlens_mini_batch

        def get_packed_packing_slice_mini_batch(packed_cu_seqlens_mini_batch, a_to_b, dp_size0, dp_size1):


            packed_packing_slice_mini_batch_send = []
            packed_packing_slice_mini_batch_recv = []
            
            for i in range(dp_size0):
                packed_packing_slice_mini_batch_send.append([0])
                pipeline_to_data_num = dict()
                for item in a_to_b[i]:
                    if item not in pipeline_to_data_num:
                        pipeline_to_data_num[item] = 0
                    pipeline_to_data_num[item] += 1
                prefix = 0
                for j in range(dp_size1):
                    if j not in pipeline_to_data_num:
                        pipeline_to_data_num[j] = 0
                    prefix += pipeline_to_data_num[j]
                    if len(packed_cu_seqlens_mini_batch[i]) > 0:
                        packed_packing_slice_mini_batch_send[i].append(packed_cu_seqlens_mini_batch[i][prefix])
                    else:
                        packed_packing_slice_mini_batch_send[i].append(0)
            
            for i in range(dp_size1):
                packed_packing_slice_mini_batch_recv.append([0])
                for j in range(dp_size0):
                    packed_packing_slice_mini_batch_recv[i].append(packed_packing_slice_mini_batch_recv[i][-1])
                    for idx, item in enumerate(a_to_b[j]):
                        if item == i:
                            packed_packing_slice_mini_batch_recv[i][-1] += packed_cu_seqlens_mini_batch[j][idx + 1] - packed_cu_seqlens_mini_batch[j][idx]

            return packed_packing_slice_mini_batch_send, packed_packing_slice_mini_batch_recv
            


        assert self._packed_vision_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_vision_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_label_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_vision_cu_seqlens_list == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_cu_seqlens_list == None, "please ensure you have not packed the bucket"
        assert self.llm_input_mask == None, "please ensure you have not packed the bucket"  
        assert self._vision_packing_slice_list == None, "please ensure you have not packed the bucket"
        assert self._llm_packing_slice_list == None, "please ensure you have not packed the bucket"

        packed_vision_input_batch = []
        packed_llm_input_batch = []
        packed_llm_label_batch = []
        packed_vision_cu_seqlens_list = []
        packed_llm_cu_seqlens_list = []
        packed_llm_input_mask = []
        vision_packing_slice_mini_batch = []
        llm_packing_slice_mini_batch = []
        vision_packing_slice_list = []
        llm_packing_slice_list = []
        
        is_visited = set()
        # 将self.batch逆序排列
        self.batch = self.batch[::-1]
        for i in range(len(self.batch)):
            packed_vision_input_mini_batch = []
            packed_vision_to_llm_postion_min_batch = []
            packed_llm_input_mini_batch = []
            packed_llm_label_mini_batch = []
            packed_vision_seqlen_mini_batch = []
            packed_llm_seqlen_mini_batch = []
            packed_llm_input_mask_mini_batch = []
            packed_llm_to_vision_llm_postion_mini_batch = []
            for j in range(vision_dp_size):
                packed_vision_input_mini_batch.append([])
                packed_vision_seqlen_mini_batch.append(0)
                packed_vision_to_llm_postion_min_batch.append([])
            for j in range(llm_dp_size):
                packed_llm_input_mini_batch.append([])
                packed_llm_label_mini_batch.append([])
                packed_llm_seqlen_mini_batch.append(0)
                packed_llm_input_mask_mini_batch.append([])
                packed_llm_to_vision_llm_postion_mini_batch.append([])

            find_data = False
            for j in range(i, len(self.batch)):
                if j in is_visited:
                    continue
                
                vision_dp_id = find_postion(j, packed_vision_seqlen_mini_batch, True)
                llm_dp_id = find_postion(j, packed_llm_seqlen_mini_batch, False)
                if vision_dp_id == -1 or llm_dp_id == -1:
                    continue
                find_data = True
                packed_vision_to_llm_postion_min_batch[vision_dp_id].append(llm_dp_id)
                packed_llm_to_vision_llm_postion_mini_batch[llm_dp_id].append(vision_dp_id)
                is_visited.add(j)
                packed_vision_input_mini_batch[vision_dp_id].append(self.batch[j].get_vision_input())
                packed_vision_seqlen_mini_batch[vision_dp_id] += self.batch[j].vision_len
                packed_llm_input_mini_batch[llm_dp_id].append(self.batch[j].get_llm_input_and_label()[0])
                packed_llm_label_mini_batch[llm_dp_id].append(self.batch[j].get_llm_input_and_label()[1])
                packed_llm_seqlen_mini_batch[llm_dp_id] += self.batch[j].text_and_video_len()
                packed_llm_input_mask_mini_batch[llm_dp_id].append(self.batch[j].get_llm_input_mask())

            if find_data:

                # 对数据进行排序
                packed_vision_input_mini_batch = sorted_lista_by_listb(packed_vision_input_mini_batch, packed_vision_to_llm_postion_min_batch)
                packed_llm_input_mini_batch = sorted_lista_by_listb(packed_llm_input_mini_batch, packed_llm_to_vision_llm_postion_mini_batch)
                packed_llm_label_mini_batch = sorted_lista_by_listb(packed_llm_label_mini_batch, packed_llm_to_vision_llm_postion_mini_batch)
                packed_llm_input_mask_mini_batch = sorted_lista_by_listb(packed_llm_input_mask_mini_batch, packed_llm_to_vision_llm_postion_mini_batch)
                packed_vision_cu_seqlens_mini_batch = get_packed_cu_seqlens_mini_batch(packed_vision_input_mini_batch, vision_dp_size)
                packed_llm_cu_seqlens_mini_batch = get_packed_cu_seqlens_mini_batch(packed_llm_input_mini_batch, llm_dp_size)
                vision_packing_slice_mini_batch, llm_packing_slice_mini_batch = get_packed_packing_slice_mini_batch(packed_vision_cu_seqlens_mini_batch, packed_vision_to_llm_postion_min_batch, vision_dp_size, llm_dp_size)
                

                for j in range(vision_dp_size):
                    if len(packed_vision_input_mini_batch[j]) == 0:
                        vision_hidden_size = self.batch[0].get_vision_input().shape[1]
                        packed_vision_input_mini_batch[j].append(np.zeros((128, vision_hidden_size), dtype=np.float32))
                        packed_vision_cu_seqlens_mini_batch[j] = [0, 128]
                    else:
                        if packed_vision_cu_seqlens_mini_batch[j][-1] % self._alignment != 0:
                            pad_seqlen = self._alignment - (packed_vision_cu_seqlens_mini_batch[j][-1] % self._alignment)
                            hidden_size = packed_vision_input_mini_batch[j][0].shape[1]
                            pad_rows = np.ones((pad_seqlen, hidden_size)) * self._pad_token
                            packed_vision_input_mini_batch[j][-1] = np.concatenate([packed_vision_input_mini_batch[j][-1], pad_rows], axis=0)
                            packed_vision_cu_seqlens_mini_batch[j][-1] = packed_vision_cu_seqlens_mini_batch[j][-1] + pad_seqlen      
                                

                    packed_vision_input_batch.append(np.concatenate(packed_vision_input_mini_batch[j]))
                    packed_vision_cu_seqlens_list.append(np.array(packed_vision_cu_seqlens_mini_batch[j], dtype=np.int32))
                    vision_packing_slice_list.append(np.array(vision_packing_slice_mini_batch[j], dtype=np.int32))

                for j in range(llm_dp_size):
                    if len(packed_llm_input_mini_batch[j]) == 0:
                        packed_llm_input_mini_batch[j].append(np.ones((128), dtype=np.int32) * self._pad_token)
                        packed_llm_label_mini_batch[j].append(np.ones((128), dtype=np.int32) * self._pad_token)
                        packed_llm_cu_seqlens_mini_batch[j] = [0, 128]
                        packed_llm_input_mask_mini_batch[j].append(np.zeros((128), dtype=np.int32))
                    else:
                        if packed_llm_cu_seqlens_mini_batch[j][-1] % self._alignment != 0:
                            pad_seqlen = self._alignment - (packed_llm_cu_seqlens_mini_batch[j][-1] % self._alignment)
                            packed_llm_input_mini_batch[j][-1] = np.concatenate([packed_llm_input_mini_batch[j][-1], np.array([self._pad_token] * pad_seqlen)])
                            packed_llm_label_mini_batch[j][-1] = np.concatenate([packed_llm_label_mini_batch[j][-1], np.array([self._pad_token] * pad_seqlen)])
                            packed_llm_cu_seqlens_mini_batch[j][-1] = packed_llm_cu_seqlens_mini_batch[j][-1] + pad_seqlen
                            packed_llm_input_mask_mini_batch[j][-1] = np.concatenate([packed_llm_input_mask_mini_batch[j][-1], np.array([0] * pad_seqlen)])

                    packed_llm_input_batch.append(np.concatenate(packed_llm_input_mini_batch[j]))
                    packed_llm_label_batch.append(np.concatenate(packed_llm_label_mini_batch[j]))
                    packed_llm_cu_seqlens_list.append(np.array(packed_llm_cu_seqlens_mini_batch[j], dtype=np.int32))
                    packed_llm_input_mask.append(np.concatenate(packed_llm_input_mask_mini_batch[j]))
                    llm_packing_slice_list.append(np.array(llm_packing_slice_mini_batch[j], dtype=np.int32))

        for i in range(len(self.batch)):
            if i not in is_visited:
                raise ValueError(f"data {i} is not visited, vision length is {self.batch[i].vision_len}, text length is {self.batch[i].text_and_video_len()}")
        

        self._packed_vision_batch = packed_vision_input_batch
        self._packed_llm_batch = packed_llm_input_batch
        self._packed_llm_label_batch = packed_llm_label_batch
        self._packed_vision_cu_seqlens_list = packed_vision_cu_seqlens_list
        self._packed_llm_cu_seqlens_list = packed_llm_cu_seqlens_list
        self.llm_input_mask = packed_llm_input_mask         
        self._llm_packing_slice_list = llm_packing_slice_list
        self._vision_packing_slice_list = vision_packing_slice_list


    def packing_method_4(self, vision_dp_size = 1, llm_dp_size = 1):


        def find_postion(data_idx : int, cu_seqlens : list, is_vision : bool):
            
            max_space = -1
            best_dp_id = -1
            for dp_id, cu_seqlen in enumerate(cu_seqlens):
                if is_vision:
                    if len(cu_seqlen) == 0:
                        remaining = self._vision_max_seqlen
                    else:
                        remaining = self._vision_max_seqlen - cu_seqlen[-1]
                    if remaining >= self.batch[data_idx].vision_len and remaining > max_space:
                        max_space = remaining
                        best_dp_id = dp_id
                else:
                    if len(cu_seqlen) == 0:
                        remaining = self._text_max_seqlen
                    else:
                        remaining = self._text_max_seqlen - cu_seqlen[-1]
                    if remaining >= self.batch[data_idx].text_and_video_len() and remaining > max_space:
                        max_space = remaining
                        best_dp_id = dp_id
            return best_dp_id



        assert self._packed_vision_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_vision_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_label_batch == None, "please ensure you have not packed the bucket"
        assert self._packed_vision_cu_seqlens_list == None, "please ensure you have not packed the bucket"
        assert self._packed_llm_cu_seqlens_list == None, "please ensure you have not packed the bucket"
        assert self.llm_input_mask == None, "please ensure you have not packed the bucket"  

        packed_vision_input_batch = []
        packed_llm_input_batch = []
        packed_llm_label_batch = []
        packed_vision_cu_seqlens_list = []
        packed_llm_cu_seqlens_list = []
        packed_llm_input_mask = []
        
        is_visited = set()
        # 将self.batch逆序排列
        self.batch = self.batch[::-1]
        for i in range(len(self.batch)):
            packed_vision_input_mini_batch = []
            packed_llm_input_mini_batch = []
            packed_llm_label_mini_batch = []
            packed_vision_cu_seqlens_mini_batch = []
            packed_llm_cu_seqlens_mini_batch = []
            packed_llm_input_mask_mini_batch = []
            for j in range(vision_dp_size):
                packed_vision_input_mini_batch.append([])
                packed_vision_cu_seqlens_mini_batch.append([])
            for j in range(llm_dp_size):
                packed_llm_input_mini_batch.append([])
                packed_llm_label_mini_batch.append([])
                packed_llm_cu_seqlens_mini_batch.append([])
                packed_llm_input_mask_mini_batch.append([])

            find_data = False
            for j in range(i, len(self.batch)):
                if j in is_visited:
                    continue
                
                vision_dp_id = find_postion(j, packed_vision_cu_seqlens_mini_batch, True)
                llm_dp_id = find_postion(j, packed_llm_cu_seqlens_mini_batch, False)
                if vision_dp_id == -1 or llm_dp_id == -1:
                    continue
                find_data = True
                is_visited.add(j)
                packed_vision_input_mini_batch[vision_dp_id].append(self.batch[j].get_vision_input())
                if len(packed_vision_cu_seqlens_mini_batch[vision_dp_id]) == 0:
                    packed_vision_cu_seqlens_mini_batch[vision_dp_id].append(0)
                    packed_vision_cu_seqlens_mini_batch[vision_dp_id].append(self.batch[j].vision_len)
                else:
                    packed_vision_cu_seqlens_mini_batch[vision_dp_id].append(packed_vision_cu_seqlens_mini_batch[vision_dp_id][-1] + self.batch[j].vision_len)
                packed_llm_input_mini_batch[llm_dp_id].append(self.batch[j].get_llm_input_and_label()[0])
                packed_llm_label_mini_batch[llm_dp_id].append(self.batch[j].get_llm_input_and_label()[1])
                if len(packed_llm_cu_seqlens_mini_batch[llm_dp_id]) == 0:
                    packed_llm_cu_seqlens_mini_batch[llm_dp_id].append(0)
                    packed_llm_cu_seqlens_mini_batch[llm_dp_id].append(self.batch[j].text_and_video_len())
                else:
                    packed_llm_cu_seqlens_mini_batch[llm_dp_id].append(packed_llm_cu_seqlens_mini_batch[llm_dp_id][-1] + self.batch[j].text_and_video_len())
                packed_llm_input_mask_mini_batch[llm_dp_id].append(self.batch[j].get_llm_input_mask())

            if find_data:
                for j in range(vision_dp_size):
                    if len(packed_vision_input_mini_batch[j]) == 0:
                        vision_hidden_size = self.batch[0].get_vision_input().shape[1]
                        packed_vision_input_mini_batch[j].append(np.zeros((128, vision_hidden_size), dtype=np.float32))
                        packed_vision_cu_seqlens_mini_batch[j] = [0, 128]
                    else:
                        if packed_vision_cu_seqlens_mini_batch[j][-1] % self._alignment != 0:
                            pad_seqlen = self._alignment - (packed_vision_cu_seqlens_mini_batch[j][-1] % self._alignment)
                            hidden_size = packed_vision_input_mini_batch[j][0].shape[1]
                            pad_rows = np.ones((pad_seqlen, hidden_size)) * self._pad_token
                            packed_vision_input_mini_batch[j][-1] = np.concatenate([packed_vision_input_mini_batch[j][-1], pad_rows], axis=0)
                            packed_vision_cu_seqlens_mini_batch[j][-1] = packed_vision_cu_seqlens_mini_batch[j][-1] + pad_seqlen      
                                

                    packed_vision_input_batch.append(np.concatenate(packed_vision_input_mini_batch[j]))
                    packed_vision_cu_seqlens_list.append(np.array(packed_vision_cu_seqlens_mini_batch[j], dtype=np.int32))

                for j in range(llm_dp_size):
                    if len(packed_llm_input_mini_batch[j]) == 0:
                        packed_llm_input_mini_batch[j].append(np.ones((128), dtype=np.int32) * self._pad_token)
                        packed_llm_label_mini_batch[j].append(np.ones((128), dtype=np.int32) * self._pad_token)
                        packed_llm_cu_seqlens_mini_batch[j] = [0, 128]
                        packed_llm_input_mask_mini_batch[j].append(np.zeros((128), dtype=np.int32))
                    else:
                        if packed_llm_cu_seqlens_mini_batch[j][-1] % self._alignment != 0:
                            pad_seqlen = self._alignment - (packed_llm_cu_seqlens_mini_batch[j][-1] % self._alignment)
                            packed_llm_input_mini_batch[j][-1] = np.concatenate([packed_llm_input_mini_batch[j][-1], np.array([self._pad_token] * pad_seqlen)])
                            packed_llm_label_mini_batch[j][-1] = np.concatenate([packed_llm_label_mini_batch[j][-1], np.array([self._pad_token] * pad_seqlen)])
                            packed_llm_cu_seqlens_mini_batch[j][-1] = packed_llm_cu_seqlens_mini_batch[j][-1] + pad_seqlen
                            packed_llm_input_mask_mini_batch[j][-1] = np.concatenate([packed_llm_input_mask_mini_batch[j][-1], np.array([0] * pad_seqlen)])

                    packed_llm_input_batch.append(np.concatenate(packed_llm_input_mini_batch[j]))
                    packed_llm_label_batch.append(np.concatenate(packed_llm_label_mini_batch[j]))
                    packed_llm_cu_seqlens_list.append(np.array(packed_llm_cu_seqlens_mini_batch[j], dtype=np.int32))
                    packed_llm_input_mask.append(np.concatenate(packed_llm_input_mask_mini_batch[j]))


        for i in range(len(self.batch)):
            if i not in is_visited:
                raise ValueError(f"data {i} is not visited")
        

        self._packed_vision_batch = packed_vision_input_batch
        self._packed_llm_batch = packed_llm_input_batch
        self._packed_llm_label_batch = packed_llm_label_batch
        self._packed_vision_cu_seqlens_list = packed_vision_cu_seqlens_list
        self._packed_llm_cu_seqlens_list = packed_llm_cu_seqlens_list
        self.llm_input_mask = packed_llm_input_mask         


    def pack_data(self, packing_method = 1, vision_dp_size = 1, llm_dp_size = 1):
        if packing_method == 0:
            raise ValueError(f"packing_method {packing_method} is not supported")
        elif packing_method == 1:
            self.packing_method_1()
        elif packing_method == 2:
            self.packing_method_2(vision_dp_size, llm_dp_size)
        elif packing_method == 3:
            self.packing_method_3(vision_dp_size, llm_dp_size)
        else:
            raise ValueError(f"packing_method {packing_method} is not supported")


    def get_packed_vision_batch(self):
        assert self._packed_vision_batch != None, "please ensure you have packed the bucket"
        return self._packed_vision_batch
    
    def get_packed_llm_batch(self):
        assert self._packed_llm_batch != None, "please ensure you have packed the bucket"
        return self._packed_llm_batch 
    
    def get_packed_llm_label_batch(self):
        assert self._packed_llm_label_batch != None, "please ensure you have packed the bucket"
        return self._packed_llm_label_batch
    
    def get_packed_vision_cu_seqlens_list(self):
        assert self._packed_vision_cu_seqlens_list != None, "please ensure you have packed the bucket"
        return self._packed_vision_cu_seqlens_list

    def get_packed_llm_cu_seqlens_list(self):
        assert self._packed_llm_cu_seqlens_list != None, "please ensure you have packed the bucket"
        return self._packed_llm_cu_seqlens_list
    
    def get_packed_llm_packing_slice_list(self):
        assert self._llm_packing_slice_list != None, "please ensure you have packed the bucket"
        return self._llm_packing_slice_list
    
    def get_packed_vision_packing_slice_list(self):
        assert self._vision_packing_slice_list != None, "please ensure you have packed the bucket"
        return self._vision_packing_slice_list


def get_sorted_by_text_and_video(global_batch: dict, pad_token: int):

    global_batch_text_len = global_batch["text_len"]
    global_batch_video_len = global_batch["video_len"]
    # 计算总长度
    total_len = global_batch_text_len + global_batch_video_len
    # 按总长度排序获取索引
    sorted_indices = np.argsort(total_len)
    
    # 对所有字段按照索引重新排序
    sorted_batch = {}
    for key in global_batch.keys():
        sorted_batch[key] = global_batch[key][sorted_indices].numpy()
        
    return sorted_batch


def get_bucket(global_batch: dict, pad_token: int, vision_max_seqlen: int, text_max_seqlen: int, mask_token: int, alignment: int):
    bucket = MultimodalBucket(pad_token, vision_max_seqlen, text_max_seqlen, alignment)
    data_num = len(global_batch["text_len"])
    for idx in range(data_num):
        bucket.add_data(global_batch["video"][idx], int(global_batch["video_len"][idx]), global_batch["text"][idx], global_batch["text_label"][idx], int(global_batch["text_len"][idx]), mask_token)

    return bucket

if __name__ == '__main__':
    pass