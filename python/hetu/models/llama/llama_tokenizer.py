from hetu.data.tokenizers.pretrained_tokenizer import PreTrainedTokenizer
from hetu.data.tokenizers.tiktoken_tokenizer import PATTERN_TIKTOKEN_CL100K

class LlamaTokenizer(PreTrainedTokenizer):
    tokenizer_class = "sentencepiece"
    
    def __init__(
        self,
        backend_tokenizer,
        **kwargs,
    ):
        kwargs.update({"tokenizer": backend_tokenizer})
        super().__init__(**kwargs)

class Llama2Tokenizer(PreTrainedTokenizer):
    tokenizer_class = "sentencepiece"
    
    def __init__(
        self,
        backend_tokenizer,
        **kwargs,
    ):
        kwargs.update({"tokenizer": backend_tokenizer})
        super().__init__(**kwargs)
    

class Llama3Tokenizer(PreTrainedTokenizer):
    tokenizer_class = "tiktoken"
    
    NUM_RESERVED_SPECIAL_TOKENS = 256
    pattern = PATTERN_TIKTOKEN_CL100K
    special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
    ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, NUM_RESERVED_SPECIAL_TOKENS - 5)
    ]
    
    def __init__(
        self,
        backend_tokenizer,
        **kwargs,
    ):
        kwargs.update({"tokenizer": backend_tokenizer})
        super().__init__(**kwargs)

        # Encode BOS and EOS, define pad ID
        self.bos_id = self.convert_tokens_to_ids("<|begin_of_text|>")
        self.eos_id = self.convert_tokens_to_ids("<|end_of_text|>")
        self.pad_id = self.convert_tokens_to_ids("<|finetune_right_pad_id|>")
        self.step_id = self.convert_tokens_to_ids("<|step_id|>")

        # Encode extra special tokens
        self.start_header_id = self.convert_tokens_to_ids("<|start_header_id|>")
        self.end_header_id = self.convert_tokens_to_ids("<|end_header_id|>")
        self.eot_id = self.convert_tokens_to_ids("<|eot_id|>")

        # During generation, stop when either eos_id, eot_id, or eom_id is encountered
        self.stop_tokens = [self.eos_id, self.eot_id]
        # TODO: refactor it to be more general
        self._tokenizer.__setattr__("pad_token", "<|finetune_right_pad_id|>")
