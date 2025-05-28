import logging
from .gpt2_tokenizer import GPT2BPETokenizer
from .sentencepiece_tokenizer import SentencePieceTokenizer
from .tiktoken_tokenizer import TikTokenizer, PATTERN_TIKTOKEN, PATTERN_TIKTOKEN_V2, PATTERN_TIKTOKEN_CL100K
from .hf_tokenizer import HuggingFaceTokenizer

pattern_dict = {
    "v1": PATTERN_TIKTOKEN,
    "v2": PATTERN_TIKTOKEN_V2,
    "cl100k": PATTERN_TIKTOKEN_CL100K,
}

def _vocab_size_with_padding(
    orig_vocab_size,
    make_vocab_size_divisible_by: int = 128,
    tensor_model_parallel_size: int = 1,
    rank: int = 0,
):
    """
    Pad vocabulary size so it is divisible by model parallel size and has a GPU-friendly size.
    
    Args:
        orig_vocab_size: Original vocabulary size.
        make_vocab_size_divisible_by: Make the vocabulary size divisible by this value.
        tensor_model_parallel_size: Tensor model parallel size.
        rank: Process rank.
        
    Returns:
        int: Padded vocabulary size.
    """
    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
    after = (after + multiple - 1) // multiple * multiple
    if rank == 0:
        logging.info(' > padded vocab (size: {}) with {} dummy tokens '
                     '(new size: {})'.format(
                         orig_vocab_size, after - orig_vocab_size, after))
    return after

def build_tokenizer(
    tokenizer_type: str,
    rank: int = 0,
    make_vocab_size_divisible_by: int = 128,
    tensor_model_parallel_size: int = 1,
    vocab_extra_ids: int = 0,
    **kwargs,
):
    """
    Initialize a tokenizer based on the specified type.
    
    Args:
        tokenizer_type: Type of tokenizer to build (e.g., 'GPT2BPETokenizer', 'SentencePieceTokenizer').
        rank: Process rank.
        make_vocab_size_divisible_by: Make the vocabulary size divisible by this value.
        tensor_model_parallel_size: Tensor model parallel size.
        vocab_extra_ids: Number of extra IDs to add to the vocabulary.
        **kwargs: Additional keyword arguments for the tokenizer.
        
    Returns:
        A tokenizer instance of the specified type.
        
    Raises:
        NotImplementedError: If the specified tokenizer type is not implemented.
    """
    if rank == 0:
        logging.info(f"> building {tokenizer_type} tokenizer ...")

    # Select and instantiate the tokenizer.
    if tokenizer_type == 'GPT2BPETokenizer':
        vocab_file = kwargs.get("vocab_file", None)
        merge_file = kwargs.get("merge_file", None)
        assert vocab_file is not None
        assert merge_file is not None
        tokenizer = GPT2BPETokenizer(vocab_file, merge_file)
    elif tokenizer_type == 'SentencePieceTokenizer':
        tokenizer_model = kwargs.get("tokenizer_model", None)
        assert tokenizer_model is not None
        tokenizer = SentencePieceTokenizer(
            tokenizer_model,
            vocab_extra_ids=vocab_extra_ids,
        )
    elif tokenizer_type == 'HuggingFaceTokenizer':
        tokenizer = HuggingFaceTokenizer(**kwargs)
    elif tokenizer_type == 'TikTokenizer':
        tokenizer_model = kwargs.get("tokenizer_model", None)
        tiktoken_pattern = kwargs.get("tiktoken_pattern", None)
        tiktoken_special_tokens = kwargs.get("tiktoken_special_tokens", None)
        assert tokenizer_model is not None
        assert tiktoken_pattern in {"v1", "v2", "cl100k"}

        pattern = pattern_dict[tiktoken_pattern]
        tokenizer = TikTokenizer(
            path=tokenizer_model,
            pattern=pattern,
            special_tokens=tiktoken_special_tokens,
        )
    else:
        raise NotImplementedError(f"{tokenizer_type} tokenizer is not implemented.")
    
    # Add vocab size.
    if kwargs.get("padded_vocab_size", None) is None:
        padded_vocab_size = _vocab_size_with_padding(
            tokenizer.vocab_size,
            make_vocab_size_divisible_by,
            tensor_model_parallel_size,
            rank,
        )
        kwargs.update({"padded_vocab_size": padded_vocab_size})
    return tokenizer
