from .utils import BaseTokenizer, SpecialToken
from typing import Dict, Any, List, Union, Iterable, Optional

WHITESPACE_CHARS = {" ", "\n", "\t", "\r", "\v"}

class SentencePieceTokenizer(BaseTokenizer, SpecialToken):
    """Tokenizer implementation based on SentencePiece.
    """
    def __init__(self, model_file, vocab_extra_ids=0, **kwargs):
        """
        Initialize the SentencePieceTokenizer.
        
        Args:
            model_file: Path to the SentencePiece model file.
            vocab_extra_ids: Number of extra IDs to add to the vocabulary.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        
        import sentencepiece
        
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initialize_special_tokens(vocab_extra_ids)
    
    def _initialize_special_tokens(self, vocab_extra_ids):
        """
        Initialize special tokens for the tokenizer.
        
        Args:
            vocab_extra_ids: Number of extra IDs to add to the vocabulary.
        """
        # add additional special tokens to special vocab
        self.special_vocab = {}
        self.inv_special_vocab = {}
        next_id = self.tokenizer.vocab_size()

        try:
            self.bos_id = self.tokenizer.bos_id()
            bos_token = self.tokenizer.id_to_piece(self.bos_id)
        except IndexError:
            bos_token = '<BOS>'
            self.bos_id = next_id
            self.special_vocab[bos_token] = next_id
            self.inv_special_vocab[next_id] = bos_token
            next_id += 1

        if not self.tokenizer.eos_id() == -1:
            self.eos_id = self.tokenizer.eos_id()
            eos_token = self.tokenizer.id_to_piece(self.eos_id)
        else:
            eos_token = '<EOS>'
            self.special_vocab[eos_token] = next_id
            self.inv_special_vocab[next_id] = eos_token
            self.eos_id = next_id
            next_id += 1
        
        if not self.tokenizer.pad_id() == -1:
            self.pad_id = self.tokenizer.pad_id()
            pad_token = self.tokenizer.id_to_piece(self.pad_id)
        else:
            pad_token = '<PAD>'
            self.special_vocab[pad_token] = next_id
            self.inv_special_vocab[next_id] = pad_token
            self.pad_id = next_id
            next_id += 1
        
        if not self.tokenizer.unk_id() == -1:
            self.unk_id = self.tokenizer.unk_id()
            unk_token = self.tokenizer.id_to_piece(self.unk_id)
        else:
            unk_token = '<UNK>'
            self.special_vocab[unk_token] = next_id
            self.inv_special_vocab[next_id] = unk_token
            self.unk_id = next_id
            next_id += 1
        
        special_tokens_dict = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
            "pad_token": pad_token,
        }
        
        self.add_special_tokens(special_tokens_dict)
        
        if self._special_tokens_map["additional_special_tokens"] is not None:
            for t in self._special_tokens_map["additional_special_tokens"]:
                self.special_vocab[t] = next_id
                self.inv_special_vocab[next_id] = t
                next_id += 1
        
        additional_special_tokens = []
        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            additional_special_tokens.append(t)
            self.special_vocab[t] = next_id
            self.inv_special_vocab[next_id] = t
            next_id += 1
            
        self._special_tokens_map["additional_special_tokens"].extend(additional_special_tokens)
        
        self.encodes_whitespace = any(
            [self.tokenizer.encode(c) for c in WHITESPACE_CHARS]
        )
    
    def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
        """
        Encode a string into a list of token ids.
        
        Args:
            text: The string to encode.
            **kwargs: Additional keyword arguments.
                add_bos: Whether to add the beginning-of-sequence token.
                add_eos: Whether to add the end-of-sequence token.
                trim_leading_whitespace: Whether to trim leading whitespace.
                prefix: A prefix to add to the text.
            
        Returns:
            A list of token ids.
        """
        add_bos = kwargs.pop("add_bos", True)
        add_eos = kwargs.pop("add_eos", False)
        trim_leading_whitespace = kwargs.pop("trim_leading_whitespace", False)
        prefix = kwargs.pop("prefix", None)
        
        if trim_leading_whitespace and self.encodes_whitespace:
            # Can define our own custom prefix depending on vocab if needed
            if not hasattr(self, "prefix"):
                self.prefix = prefix or "\n"
                self.encoded_prefix = self._inner_encode(self.prefix, add_bos=False, add_eos=False)
            start_idx = len(self.encoded_prefix) + int(add_bos)
            return self._inner_encode(text, add_bos, add_eos)[start_idx:]
        else:
            return self._inner_encode(text, add_bos, add_eos)
    
    def _inner_encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        Inner implementation of encoding a string to token ids.
        
        Args:
            text: The text to encode.
            add_bos: Whether to add the beginning-of-sequence token.
            add_eos: Whether to add the end-of-sequence token.
            
        Returns:
            A list of token ids.
        """
        if self.special_vocab is None:
            return self.tokenizer.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                out_type=int
            )
        else:
            ids = []
            idx = 0
            
            while 1:
                indices = {}
                for token in self.special_vocab:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue
                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
                ids.append(self.special_vocab[next_token])
                idx = next_idx + len(next_token)

            ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids.append(self.eos_id)
            return ids

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        **kwargs: Dict[str, Any]
    ) -> str:
        """
        Decode a list of token ids into a string.
        
        Args:
            token_ids: The token ids to decode.
            **kwargs: Additional keyword arguments.
        
        Returns:
            The decoded string.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if self.inv_special_vocab is None:
            return self.tokenizer.decode(token_ids)
        else:
            text = ""
            last_i = 0

            for i, id in enumerate(token_ids):
                if id in self.inv_special_vocab:
                    text += self.tokenizer.decode_ids(token_ids[last_i:i]) + " "
                    text += self.inv_special_vocab[id] + " "
                    last_i = i + 1

            text += self.tokenizer.decode_ids(token_ids[last_i:])
            return text

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
        """
        Convert token ids to tokens.
        
        Args:
            ids: The token ids to convert.
        
        Returns:
            The corresponding tokens.
        """
        try:
            return self.tokenizer.id_to_piece(ids)
        except:
            if isinstance(ids, int):
                return self.inv_special_vocab[ids]
            else:
                return [self.inv_special_vocab[id] for id in ids]
    
    def convert_tokens_to_ids(self, tokens: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        """
        Convert tokens to token ids.
        
        Args:
            tokens: The tokens to convert.
        
        Returns:
            The corresponding token ids.
        """
        try:
            return self.tokenizer.piece_to_id(tokens)
        except:
            if isinstance(tokens, str):
                return self.special_vocab[tokens]
            else:
                return [self.special_vocab[t]for t in tokens]
    
    @property
    def vocab_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            The size of the vocabulary.
        """
        return self.tokenizer.vocab_size()
