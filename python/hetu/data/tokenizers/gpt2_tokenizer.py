import sys
import json
import logging
import os
import regex as re
from io import open
from .utils import BaseTokenizer
from typing import Dict, Any, List, Union, Iterable, Optional

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE
    # tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func

logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
}
PRETRAINED_MERGES_ARCHIVE_MAP = {
    'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'gpt2': 1024,
}
VOCAB_NAME = 'vocab.json'
MERGES_NAME = 'merges.txt'
SPECIAL_TOKENS_NAME = 'special_tokens.txt'

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And we avoid mapping to whitespace/control characters that the bpe code barfs on.

    Returns:
        dict: A dictionary mapping from byte values to unicode strings.
    """
    _chr = unichr if sys.version_info[0] == 2 else chr
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).

    Args:
        word (tuple): A tuple of symbols representing a word.
        
    Returns:
        set: A set of tuples containing symbol pairs found in the word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class GPT2Tokenizer(object):
    """
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level BPE
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.

        Args:
            pretrained_model_name_or_path (str): path to a local file.
            cache_dir (str, optional): Path to store cached models in.
            *inputs: Additional positional arguments to pass to the tokenizer constructor.
            **kwargs: Additional keyword arguments to pass to the tokenizer constructor.
            
        Returns:
            GPT2Tokenizer: A GPT-2 tokenizer instance.
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
            merges_file = PRETRAINED_MERGES_ARCHIVE_MAP[pretrained_model_name_or_path]
            special_tokens_file = None
        else:
            vocab_file = os.path.join(pretrained_model_name_or_path, VOCAB_NAME)
            merges_file = os.path.join(pretrained_model_name_or_path, MERGES_NAME)
            special_tokens_file = os.path.join(pretrained_model_name_or_path, SPECIAL_TOKENS_NAME)
            if not os.path.exists(special_tokens_file):
                special_tokens_file = None
            else:
                logger.info("loading special tokens file {}".format(special_tokens_file))
        # redirect to the cache, if necessary
        try:
            from hetu.utils.file_utils import cached_path
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
            resolved_merges_file = cached_path(merges_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find files {} and {} "
                "at this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    pretrained_model_name_or_path,
                    vocab_file, merges_file))
            return None
        if resolved_vocab_file == vocab_file and resolved_merges_file == merges_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
            logger.info("loading merges file {}".format(merges_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
            logger.info("loading merges file {} from cache at {}".format(
                merges_file, resolved_merges_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        if special_tokens_file and 'special_tokens' not in kwargs:
            special_tokens = open(special_tokens_file, encoding='utf-8').read().split('\n')[:-1]
        else:
            special_tokens = kwargs.pop('special_tokens', [])
        tokenizer = cls(
            resolved_vocab_file,
            resolved_merges_file,
            special_tokens=special_tokens,
            *inputs,
            **kwargs)
        return tokenizer

    def __init__(self, vocab_file, merges_file, errors='replace',
                 special_tokens=None, max_len=None, **kwargs):
        """
        Initialize a GPT-2 tokenizer instance.
        
        Args:
            vocab_file (str): Path to the vocabulary file.
            merges_file (str): Path to the merges file.
            errors (str, optional): How to handle encoding errors. Defaults to 'replace'.
            special_tokens (List[str], optional): List of special tokens. Defaults to None.
            max_len (int, optional): Maximum length of input tokens. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for
        # capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)

    def __len__(self):
        """
        Get the size of the vocabulary.
        
        Returns:
            int: The sum of the base vocabulary size and the number of special tokens.
        """
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens):
        """
        Add a list of additional tokens to the encoder.
        The additional tokens are indexed starting from the last index of the
        current vocabulary in the order of the 'special_tokens' list.

        Args:
            special_tokens (List[str]): The list of special tokens to add.
            
        Returns:
            None
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i)
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        logger.info("Special tokens {}".format(self.special_tokens))

    def bpe(self, token):
        """
        Perform Byte-Pair Encoding on a token.
        
        Args:
            token (str): The token to encode using BPE.
            
        Returns:
            str: The BPE-encoded token.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except BaseException:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """
        Tokenize a string into BPE tokens.
        
        Args:
            text (str): The string to tokenize.
            
        Returns:
            List[str]: A list of BPE tokens.
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(self.byte_encoder[ord(b)] for b in token)
            else:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        """
        Convert a token or list of tokens to their corresponding IDs in the vocabulary.
        
        Args:
            tokens (Union[str, Iterable[str]]): A token or list of tokens to convert.
            
        Returns:
            Union[int, List[int]]: The ID or list of IDs corresponding to the token(s).
            
        Warns:
            If the resulting token sequence is longer than the maximum allowed length.
        """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(
                    len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """
        Convert a sequence of IDs to their corresponding tokens.
        
        Args:
            ids (List[int]): A list of token IDs to convert.
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to False.
            
        Returns:
            List[str]: A list of tokens corresponding to the input IDs.
        """
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text):
        """
        Encode a string into token IDs.
        
        Args:
            text (str): The string to encode.
            
        Returns:
            List[int]: A list of token IDs.
        """
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, tokens):
        """
        Decode a list of token IDs back to a string.
        
        Args:
            tokens (List[int]): A list of token IDs to decode.
            
        Returns:
            str: The decoded string.
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def save_vocabulary(self, vocab_path):
        """
        Save the tokenizer vocabulary and merge files to a directory.
        
        Args:
            vocab_path (str): The directory where to save the vocabulary.
            
        Returns:
            Tuple[str, str, str]: Paths to the saved vocabulary, merges, and special tokens files.
            
        Warns:
            If BPE merge indices are not consecutive, which might indicate a corrupted tokenizer.
        """
        if not os.path.isdir(vocab_path):
            logger.error("Vocabulary path ({}) should be a directory".format(vocab_path))
            return
        vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        merge_file = os.path.join(vocab_path, MERGES_NAME)
        special_tokens_file = os.path.join(vocab_path, SPECIAL_TOKENS_NAME)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write(u'#version: 0.2\n')
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        index = len(self.encoder)
        with open(special_tokens_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.special_tokens.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving special tokens vocabulary to {}: BPE indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(special_tokens_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1

        return vocab_file, merge_file, special_tokens_file


class GPT2BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_file, merge_file):
        """
        Initialize a GPT-2 BPE tokenizer that implements the BaseTokenizer interface.
        
        Args:
            vocab_file (str): Path to the vocabulary file.
            merge_file (str): Path to the merges file.
        """
        super().__init__()
        self.tokenizer = GPT2Tokenizer(vocab_file, merge_file, errors='replace',
                                       special_tokens=['<pad>'], max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']
        self.pad_id = self.tokenizer.special_tokens['<pad>']
    
    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    @property
    def eod(self):
        return self.eod_id
    
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
        return self.tokenizer.encode(text)

    def _decode(self, token_ids: Union[int, List[int]], **kwargs: Dict[str, Any]) -> str:
        return self.tokenizer.decode(token_ids)
    
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
