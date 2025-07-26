from hetu.data.tokenizers.pretrained_tokenizer import PreTrainedTokenizer

class GPTTokenizer(PreTrainedTokenizer):
    tokenizer_class = "gpt2"
    
    def __init__(
        self,
        backend_tokenizer,
        **kwargs,
    ):
        kwargs.update({"tokenizer": backend_tokenizer})
        super().__init__(**kwargs)
