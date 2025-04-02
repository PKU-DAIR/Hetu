from .utils import BaseTokenizer
from typing import Dict, Any, List, Union, Optional, Iterable
from pathlib import Path

PATTERN_TIKTOKEN = (
    r"[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)
PATTERN_TIKTOKEN_V2 = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

PATTERN_TIKTOKEN_CL100K = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Constants controlling encode logic
MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACE_CHARS = 25_000

class TikTokenizer(BaseTokenizer):
    def __init__(
        self,
        path: str,
        pattern: str,
        special_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        
        from tiktoken import Encoding
        from tiktoken.load import load_tiktoken_bpe
        
        default_special_tokens = ["<s>", "</s>", "<unk>"]
        if special_tokens is None:
            special_tokens = default_special_tokens

        mergeable_ranks = load_tiktoken_bpe(path)
        num_base_tokens = len(mergeable_ranks)
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        
        # default setting
        self.bos_id = self.special_tokens.get("<s>", None)
        self.eos_id = self.special_tokens.get("</s>", None)
        self.pad_id = self.eos_id

        self.tt_model = Encoding(
            name=Path(path).parent.name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        
        if self.bos_id is None and self.eos_id is None:
            print(f"Warning: special token ids not set, please set them manually in specific models.")

    def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
        add_bos = kwargs.pop("add_bos", True)
        add_eos = kwargs.pop("add_eos", True)

        substrs: List[str] = []
        tokens = []
        if not text:
            return []
        for i in range(0, len(text), MAX_ENCODE_CHARS):
            substr = text[i : i + MAX_ENCODE_CHARS]
            # See https://github.com/openai/tiktoken/issues/195
            sliced_substr = self._split_long_repetitions(
                substr, MAX_NO_WHITESPACE_CHARS
            )
            substrs.extend(sliced_substr)
        for substr in substrs:
            tokens.extend(
                self.tt_model.encode(
                    substr,
                    disallowed_special=(),
                )
            )
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        **kwargs: Dict[str, Any]
    ) -> str:
        truncate_at_eos = kwargs.pop("truncate_at_eos", False)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if truncate_at_eos:
            try:
                k = token_ids.index(self.eos_id)
            except ValueError:
                k = None
            if k:
                token_ids = token_ids[:k]
        return self.tt_model.decode(token_ids)

    def tokenize_messages(
        self,
        messages,
        prompt_template: Optional[str] = None,
        tokenize: bool = True,
        return_mask: bool = True,
        return_dict: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Union[str, List[int], Dict[str, List[int]]]:
        # TODO: Implement it
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        return self.tt_model.decode(ids)

    def convert_tokens_to_ids(self, tokens: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        return self.tt_model.encode(tokens)
    
    @property
    def vocab_size(self) -> int:
        return self.tt_model.n_vocab

    @property
    def base_vocab_size(self) -> int:
        return self.vocab_size - len(self.special_tokens)
