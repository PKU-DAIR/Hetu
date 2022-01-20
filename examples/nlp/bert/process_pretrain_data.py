from datasets import load_dataset
import random
import hetu
import os
import numpy as np

''' Usage example:
    In dir Hetu/examples/nlp/bert/: python processBertData.py
'''

# https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [str(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)
    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))
    masked_lms = []
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        masked_token = None
        # replace with [MASK] at 80%.
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # keep original at 10%.
            if rng.random() < 0.5:
                masked_token = tokens[index]
            #  replace with random word at 10%.
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
        output_tokens[index] = masked_token
        masked_lms.append([index, tokens[index]])

    masked_lms.sort(key = lambda x: x[0])
    masked_lm_positions = []
    masked_lm_labels = []

    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_data_from_document(all_document,  doc_id, max_seq_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """ Create Training example for input document """
    document = all_document[doc_id]
    max_num_tokens = max_seq_length - 3 # [CLS], [SEP], [SEP]
    target_seq_length = max_num_tokens
    # generate short sequence at the probility of short_seq_prob
    # In order to minimize the mismatch between pre-training and fine-tuning.
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # create sentence A
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend([current_chunk[j]])
                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_document) - 1)
                        if random_document_index != doc_id:
                            break
                    #If picked random document is the same as the current document
                    if random_document_index == doc_id:
                        is_random_next = False
                    random_document = all_document[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend([random_document[j]])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend([current_chunk[j]])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances
    
def convert_instances_to_data(instances, tokenizer, max_seq_length):
    
    num_instances = len(instances)
    input_ids_list = np.zeros([num_instances, max_seq_length], dtype="int32")
    input_mask_list = np.zeros([num_instances, max_seq_length], dtype="int32")
    segment_ids_list = np.zeros([num_instances, max_seq_length], dtype="int32")
    masked_lm_labels = np.full([num_instances, max_seq_length],-1, dtype="int32")
    next_sentence_labels_list = np.zeros(num_instances, dtype="int32")

    for (idx, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        padding_zero_list = [0]*int(max_seq_length - len(input_ids))
        input_ids += padding_zero_list
        input_mask += padding_zero_list
        segment_ids += padding_zero_list

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)

        input_ids_list[idx][:] = input_ids
        input_mask_list[idx][:] = input_mask
        segment_ids_list[idx][:] = segment_ids
        masked_lm_labels[idx][masked_lm_positions] = masked_lm_ids
        next_sentence_labels_list[idx] = 1 if instance.is_random_next else 0

    return input_ids_list, input_mask_list, segment_ids_list, masked_lm_labels, next_sentence_labels_list

def create_pretrain_data(dataset, tokenizer, max_seq_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq, rng):
    documents,  all_data = [], [[],[],[],[],[]]
    vocab_words = list(tokenizer.vocab.keys())

    save_path='./preprocessed_data/bookcorpus/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(dataset['train'].shape[0]):
        tokens = tokenizer.tokenize(dataset['train'][i]['text'])
        documents.append(tokens)
        instance = create_data_from_document(documents, i,\
                            max_seq_length, short_seq_prob, masked_lm_prob, 
                            max_predictions_per_seq, vocab_words, rng)
        data = convert_instances_to_data(instance, tokenizer, max_seq_length)
        print(i, len(tokens), len(instance))
        for j in range(5):
            all_data[j].append(data[j]) 

        save_gap=200
        if (i+1)%save_gap==0 and i:
            input_ids_list, input_mask_list, segment_ids_list, masked_lm_labels, next_sentence_labels_list = [np.concatenate(all_data[j],axis=0) for j in range(5)]
            print('Saving data from %d to %d: doc_num = %d, input_ids_shape ='%(i+1-save_gap,i, i+1), input_ids_list.shape)
            save_data(input_ids_list, input_mask_list, segment_ids_list, masked_lm_labels, next_sentence_labels_list, name='_%d_%d'%(i+1-save_gap,i))
            all_data = [[],[],[],[],[]]
        if i == dataset['train'].shape[0]-1:
            input_ids_list, input_mask_list, segment_ids_list, masked_lm_labels, next_sentence_labels_list = [np.concatenate(all_data[j],axis=0) for j in range(5)]
            print('Saving data from %d to %d: doc_num = %d, input_ids_shape ='%(save_gap*int(i/save_gap),i, i+1), input_ids_list.shape)
            save_data(input_ids_list, input_mask_list, segment_ids_list, masked_lm_labels, next_sentence_labels_list, name='_%d_%d'%(save_gap*int(i/save_gap),i))

def save_data(input_ids_list, input_mask_list, segment_ids_list, masked_lm_labels, next_sentence_labels_list,name=''):
    save_path='./preprocessed_data/bookcorpus/'
    np.save(save_path+'input_ids'+name,np.array(input_ids_list))
    np.save(save_path+'token_type_ids'+name,np.array(segment_ids_list))
    np.save(save_path+'attention_mask'+name,np.array(input_mask_list))
    np.save(save_path+'masked_lm_labels'+name,np.array(masked_lm_labels))
    np.save(save_path+'next_sentence_label'+name,np.array(next_sentence_labels_list))

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        #add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def show_dataset_detail(dataset):
    print(dataset.shape)
    print(dataset.column_names)
    print(dataset['train'].features)
    print(dataset['train'][0]['text'])

if __name__ == "__main__":
    max_seq_length = 512
    do_lower_case = True
    short_seq_prob = 0.1
    masked_lm_prob = 0.15
    max_predictions_per_seq = 20
    
    vocab_path = "./datasets/bert-base-uncased-vocab.txt"
    dataset = load_dataset('../bookcorpus', cache_dir = "./cached_data")
    
    print("total number of documents {} ".format(dataset['train'].shape[0]))

    random_seed = 123
    rng = random.Random(random_seed)
    tokenizer = hetu.BertTokenizer(vocab_file=vocab_path, do_lower_case = do_lower_case)

    print("vocab_size =",len(tokenizer.vocab))
    print("max_seq_len =", max_seq_length)
    
    create_pretrain_data(dataset, tokenizer, max_seq_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq, rng)





