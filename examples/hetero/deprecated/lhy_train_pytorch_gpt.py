from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.optim import SGD
import ptvsd

# ptvsd.enable_attach(address =('127.0.0.1', 4000))
# ptvsd.wait_for_attach()
tokenizer = GPT2Tokenizer.from_pretrained('./checkpoint/HuggingFace')
model = GPT2LMHeadModel.from_pretrained('./checkpoint/HuggingFace')

encoded_inputs = []
text = ['Hello, I am',
        "Good morning! Today",
        "There is a question",
        "Where can I find",
        'Hello, I am',
        "Good morning! Today",
        "There is a question",
        "Where can I find"]
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input['labels'] = encoded_input['input_ids']
encoded_inputs.append(encoded_input)
text = ['Hello, I am a',
        "Good morning! Today is",
        "There is a question about",
        "Where can I find the",
        'Cool, you are so',
        "Well done! The thing",
        "I have so many things",
        "Why the child is going",
        'I like playing tennis and',
        "Do you think this apple",
        "This sounds good for me",
        "Have you ever been to",
        'It is my pleasure to',
        "I think people are all",
        "Is there any chance that",
        "We are good friends and"]
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input['labels'] = encoded_input['input_ids']
encoded_inputs.append(encoded_input)
'''
text = ['Hello, I am a good',
        "Good morning! Today is a",
        "There is a question about whether",
        "Where can I find the best"]
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input['labels'] = encoded_input['input_ids']
encoded_inputs.append(encoded_input)
text = ['Hello, I am a',
        "Good morning! Today is",
        "There is a question about",
        "Where can I find the"]
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input['labels'] = encoded_input['input_ids']
encoded_inputs.append(encoded_input)
'''

model.train()
opt = SGD(model.parameters(), lr=0.01)

for i in range(len(encoded_inputs)):
        output = model(**encoded_inputs[i])
        loss = output.loss
        print('loss:', loss, loss.dtype)
        '''
        lm_logits = output.logits[:, -1, :]
        print('lm_logits:', lm_logits)
        new_ids = lm_logits.argmax(dim=-1)
        print('new_ids:', new_ids, 'new_tokens:', tokenizer.decode(new_ids))
        '''
        opt.zero_grad()
        loss.backward()
        opt.step()
        
torch.save(model.state_dict(), './checkpoint/temp/hf_pytorch_model.bin')