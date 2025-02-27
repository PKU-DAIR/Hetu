import numpy as np
import os
import pickle
import h5py


class DataLoaderForGPT(object):
    def __init__(self, input_file, batch_size, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.batch_size = batch_size
        self.data_names = ['input_ids','token_type_ids','attention_mask','masked_lm_positions', 'masked_lm_ids', 'next_sentence_label']
        self.data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'masked_lm_positions':[],
                    'masked_lm_ids':[],
                    'next_sentence_label':[]}
        self.batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'masked_lm_positions':[],
                    'masked_lm_ids':[],
                    'next_sentence_label':[]}
        self.cur_batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'masked_lm_positions':[],
                    'masked_lm_ids':[],
                    'next_sentence_label':[]}
        f = h5py.File(input_file, "r")
        hdf5_keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.data['input_ids'], self.data['attention_mask'], self.data['token_type_ids'], \
            self.data['masked_lm_positions'], self.data['masked_lm_ids'], \
                self.data['next_sentence_label'] = [np.asarray(f[key][:]) for key in hdf5_keys]
        f.close()
        self.data_len = self.data['input_ids'].shape[0]

        print('Successfully loaded data file %s!'%input_file)
        self.make_epoch_data()

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

        input_ids = self.cur_batch_data['input_ids']
        masked_lm_positions = self.cur_batch_data['masked_lm_positions']
        masked_lm_ids = self.cur_batch_data['masked_lm_ids']

        masked_lm_labels = np.ones(input_ids.shape, dtype=np.int64) * -1

        for i in range(masked_lm_positions.shape[0]):
            index = self.max_pred_length
            # store number of  masked tokens in index
            padded_mask_indices = (masked_lm_positions[i] == 0).nonzero()
            if len(padded_mask_indices[0]) != 0:
                index = padded_mask_indices[0][0]
            masked_lm_labels[i][masked_lm_positions[i][:index]] = masked_lm_ids[i][:index]

        self.cur_batch_data['masked_lm_labels'] = masked_lm_labels

        return self.cur_batch_data.copy()


class DataLoaderForGlue(object):
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
        self.make_epoch_data()

    def load_data(self, task_name='sst-2', datatype='train'):
        print('Loading preprocessed dataset %s...'%task_name)
        cached_train_features_file = os.path.join('./data/preprocessed_glue_data/','%s_%s_features'%(task_name,datatype))

        try:
            with open(cached_train_features_file, "rb") as reader:
                self.data = pickle.load(reader)
            print("Loaded pre-processed features from {}".format(
                cached_train_features_file))
        except:
            print("Did not find pre-processed features from {}".format(
                cached_train_features_file))
            print("Please run sh scripts/create_datasets_from_start.sh first!")
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