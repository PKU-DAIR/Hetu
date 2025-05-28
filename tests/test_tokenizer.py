from hetu import Llama3Tokenizer
from transformers import LlamaTokenizerFast

def test_llama3_tokenizer():
    hetu_tokenizer = Llama3Tokenizer.from_pretrained("/path/to/llama3/tokenizer")
    hf_tokenizer = LlamaTokenizerFast.from_pretrained("/path/to/llama3/tokenizer")
    
    assert hetu_tokenizer.encode("Hello this is a test") == hf_tokenizer.encode("Hello this is a test")
    assert hetu_tokenizer.eos_token_id == hf_tokenizer.eos_token_id
    assert hetu_tokenizer.eos_id == hf_tokenizer.eos_token_id
    assert hetu_tokenizer.decode([128001]) == hf_tokenizer.decode([128001])

if "__main__" == __name__:
    test_llama3_tokenizer()