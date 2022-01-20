# BERT Example
In this directory we provide implementations for BERT model on PyTorch and Hetu.
## CCFBDCI 2021
CCFBDCI 2021 (BERT memory optimization competition) requires us to train BERT Large model on 2 GPUs (12GB each) with hidden_size as large as possible.  
We provide reproduction of all techniques we used in CCFBDCI 2021, including pipeline parallel scheme (GPipe) on 2 GPUs, gradient applied immediately, operation fusion, dynamic memory management. As a result, on Hetu we can train BERT Large model with hidden_size up to 2128.  
In datasets directory, we provide example data for BERT training, which contains 2048 sequences.  
To run hidden_size = 2128 version of BERT Large model on hetu, run following scripts:  
```bash
sh run_hetu_bert_ccfbdci_2128.sh
```