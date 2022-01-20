import numpy as np
import os
import pickle
import h5py

class DataLoader(object):
    def __init__(self, dataset='bookcorpus', doc_num=16000, save_gap=200, batch_size = 1024):
        self.data_names = ['input_ids','token_type_ids','attention_mask','masked_lm_labels','next_sentence_label']
        self.data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'masked_lm_labels':[],
                    'next_sentence_label':[]}
        self.batch_size=batch_size
        self.batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'masked_lm_labels':[],
                    'next_sentence_label':[]}
        self.cur_batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'masked_lm_labels':[],
                    'next_sentence_label':[]}
        self.load_data(dataset=dataset, doc_num=doc_num, save_gap=save_gap)


    def load_data(self, dataset='bookcorpus', doc_num=16000, save_gap=200):
        print('Loading preprocessed dataset %s...'%dataset)
        data_dir = './preprocessed_data/%s/'%dataset

        for i in range(0,doc_num,save_gap):
            start, end = i, i+save_gap-1
            if end > doc_num-1:
                end = doc_num-1
            range_name = '_%d_%d.npy'%(start,end)
            print(start,end)
            for data_name in self.data_names:
                #print(data_dir+data_name+range_name)
                self.data[data_name].append(np.load(data_dir+data_name+range_name))
        
        for data_name in self.data_names:
            self.data[data_name] = np.concatenate(self.data[data_name],axis=0)
        
        self.data_len = self.data['input_ids'].shape[0]
        print(self.data['input_ids'].shape)

        print('Successfully loaded dataset %s!'%dataset)


    def make_epoch_data(self):
        for i in range(0, self.data_len, self.batch_size):
            start = i
            end = start + self.batch_size
            if end > self.data_len:
                end = self.data_len
            if end-start != self.batch_size:
                break
            for data_name in self.data_names:
                self.batch_data[data_name].append(self.data[data_name][start:end]) 

        self.batch_num = len(self.batch_data['input_ids'])
    
    def get_batch(self, idx):
        if idx >= self.batch_num:
            assert False
        for data_name in self.data_names:
            self.cur_batch_data[data_name] = self.batch_data[data_name][idx]

        return self.cur_batch_data.copy()
    
    def align(self, arr, length):
        ori_len = len(arr)
        if length > ori_len:
            return arr + [0] * (length - ori_len)
        else:
            return arr[:length]


class DataLoader_example_data_for_bert(object):
    def __init__(self, batch_size):
        self.data_len = 2048
        self.batch_size = batch_size
        self.data_names = ['input_ids','token_type_ids','attention_mask','masked_lm_labels','next_sentence_label']
        self.data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'masked_lm_labels':[],
                    'next_sentence_label':[]}
        self.batch_size=batch_size
        self.batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'masked_lm_labels':[],
                    'next_sentence_label':[]}
        self.cur_batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'masked_lm_labels':[],
                    'next_sentence_label':[]}
        f = h5py.File('./datasets/example_data_for_bert.hdf5', "r")
        for data_name in self.data_names: 
            self.data[data_name] = np.asarray(f[data_name][:])
        f.close()

    def make_epoch_data(self):
        for i in range(0, self.data_len, self.batch_size):
            start = i
            end = start + self.batch_size
            if end > self.data_len:
                end = self.data_len
            if end-start != self.batch_size:
                break
            for data_name in self.data_names:
                self.batch_data[data_name].append(self.data[data_name][start:end]) 

        self.batch_num = len(self.batch_data['input_ids'])
    
    def get_batch(self, idx):
        if idx >= self.batch_num:
            assert False
        for data_name in self.data_names:
            self.cur_batch_data[data_name] = self.batch_data[data_name][idx]

        return self.cur_batch_data.copy()
    

class DataLoader4Glue(object):
    def __init__(self, task_name='sst-2', batch_size = 1024, datatype='train'):
        self.data_names = ['input_ids','token_type_ids','attention_mask','label_ids']
        self.data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'label_ids':[]}
        self.batch_size=batch_size
        self.batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'label_ids':[]}
        self.cur_batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'label_ids':[]}
        self.load_data(task_name=task_name, datatype=datatype)

    def load_data(self, task_name='sst-2', datatype='train'):
        print('Loading preprocessed dataset %s...'%task_name)
        cached_train_features_file = os.path.join('./preprocessed_data/glue','%s_%s_features'%(task_name,datatype),)

        try:
            with open(cached_train_features_file, "rb") as reader:
                self.data = pickle.load(reader)
            print("Loaded pre-processed features from {}".format(
                cached_train_features_file))
        except:
            print("Did not find pre-processed features from {}".format(
                cached_train_features_file))
            print("Please run process_glue_data.py first!")
            assert False

        self.data_len = self.data['input_ids'].shape[0]
        self.num_labels = np.max(self.data['label_ids'])+1
        print(self.data['input_ids'].shape)
        print('Successfully loaded GLUE dataset %s for %s!'%(task_name,datatype))
    
    def make_epoch_data(self):
        for i in range(0, self.data_len, self.batch_size):
            start = i
            end = start + self.batch_size
            if end > self.data_len:
                end = self.data_len
            if end-start != self.batch_size:
                break
            for data_name in self.data_names:
                self.batch_data[data_name].append(self.data[data_name][start:end]) 

        self.batch_num = len(self.batch_data['input_ids'])
    
    def get_batch(self, idx):
        if idx >= self.batch_num:
            assert False
        for data_name in self.data_names:
            self.cur_batch_data[data_name] = self.batch_data[data_name][idx]

        return self.cur_batch_data.copy()