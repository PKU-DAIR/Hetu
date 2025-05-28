import os
import copy
import json
import logging
from .utils import BaseTokenizer, SPECIAL_TOKENS_ATTRIBUTE, PaddingStrategy, TruncationStrategy
import tokenizers.pre_tokenizers as pre_tokenizers
from tokenizers import Tokenizer, AddedToken, Encoding
from tokenizers.decoders import Decoder
from typing import Any, Dict, List, Union, Optional, Iterable, Tuple, Callable
from hetu.data.messages.prompt_template import PromptTemplate

INIT_MODEL_MAX_LEN = int(1e30)
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
PROMPT_TEMPLATE_FILE = "chat_template.jinja"
TOKENIZER_FILE = "tokenizer.json"

class HuggingFaceSpecialToken:
    special_tokens_attribute = SPECIAL_TOKENS_ATTRIBUTE

    def __init__(self, **kwargs):
        self.pad_token_type_id = 0
        self._special_tokens_map = {attr: None for attr in self.special_tokens_attribute}
        self._special_tokens_map["additional_special_tokens"] = []
        
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.special_tokens_attribute:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(
                        isinstance(t, (str, AddedToken)) for t in value
                    ), "One of the tokens is not a string or an AddedToken"
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"Special token {key} has to be either str or AddedToken but got: {type(value)}")

    def add_special_tokens(
        self, special_tokens_dict: Dict[str, Union[str, AddedToken]], override_additional_special_tokens=True
    ) -> int:
        if not special_tokens_dict:
            return 0
        
        added_tokens = []
        for key, value in special_tokens_dict.items():
            assert key in self.special_tokens_attribute, \
            f"Key {key} is not a special token. If you want to add a new special token,"
            " please set key to 'additional_special_tokens' and provide a list of str or AddedToken"

            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, (str, AddedToken)) for t in value
                ), f"Tokens {value} for key {key} should all be str or AddedToken instances"

                to_add = []
                for token in value:
                    if isinstance(token, str):
                        # for legacy purpose we default to stripping. `test_add_tokens_tokenizer` depends on this
                        token = AddedToken(token, rstrip=False, lstrip=False, normalized=False, special=True)
                    if not override_additional_special_tokens and str(token) in self.additional_special_tokens:
                        continue
                    to_add.append(token)
                if override_additional_special_tokens and len(to_add) > 0:
                    setattr(self, key, list(to_add))
                else:
                    self._special_tokens_map["additional_special_tokens"].extend(to_add)
                added_tokens += to_add
            else:
                if not isinstance(value, (str, AddedToken)):
                    raise ValueError(f"Token {value} for key {key} should be a str or an AddedToken instance")
                if isinstance(value, (str)):
                    # for legacy purpose we default to stripping. `False` depends on this
                    value = AddedToken(value, rstrip=False, lstrip=False, normalized=False, special=True)
                setattr(self, key, value)
                if value not in added_tokens:
                    added_tokens.append(value)

        # if we are adding tokens that were not part of the vocab, we ought to add them
        added_tokens = self.add_tokens(added_tokens, special_tokens=True)
        return added_tokens

    def add_tokens(
        self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False
    ) -> int:
        if not new_tokens:
            return 0
        
        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]
        
        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise NotImplementedError
    
    def __setattr__(self, key, value):
        # can pass in tokens/ids
        key_without_id = key
        key_is_id = key.endswith("_id") or key.endswith("_ids")
        if key_is_id:
            key_without_id = key[:-3] if not key.endswith("_ids") else key[:-4]
        
        if key_is_id and not key_without_id.endswith("_token"):
            key_without_id += "_token"

        if self.__dict__.get("_special_tokens_map", None) is not None and any(
            name in self.__dict__["_special_tokens_map"] for name in [key, key_without_id]
        ):
            if key_is_id:
                if value is not None:
                    value = (
                        self.convert_ids_to_tokens(value)
                        if key != "additional_special_tokens"
                        else [self.convert_ids_to_tokens(val) for val in value]
                    )
                key = key_without_id

            if key != "additional_special_tokens" and not isinstance(value, (str, AddedToken)) and value is not None:
                raise ValueError(f"Cannot set a non-string value as the {key}")
            self._special_tokens_map[key] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        # can get tokens/ids
        key_without_id = key
        key_is_id = key.endswith("_id") or key.endswith("_ids")
        if key_is_id:
            key_without_id = key[:-3] if not key.endswith("_ids") else key[:-4]
        
        if key_is_id and not key_without_id.endswith("_token"):
            key_without_id += "_token"

        if self.__dict__.get("_special_tokens_map", None) is not None and any(
            name in self.__dict__["_special_tokens_map"] for name in [key, key_without_id]
        ):
            _special_tokens_map = self.__dict__["_special_tokens_map"]
            if not key_is_id:
                if _special_tokens_map[key] is None:
                    return None
                value = _special_tokens_map[key]
                return str(value) if key != "additional_special_tokens" else [str(tok) for tok in value]
            else:
                attr_as_tokens = getattr(self, key_without_id)
                return self.convert_tokens_to_ids(attr_as_tokens) if attr_as_tokens is not None else None
        elif self.__dict__.get("_special_tokens_map", None) is not None and any(
            name in self.__dict__["_special_tokens_map"]["additional_special_tokens"] for name in [key, key_without_id]
        ):
            _additional_special_tokens = self.__dict__["_special_tokens_map"]["additional_special_tokens"]
            if key_is_id:
                for token in _additional_special_tokens:
                    if token == key_without_id:
                        return self.convert_tokens_to_ids(token)
            else:
                return key

        if key not in self.__dict__:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
        else:
            return super().__getattr__(key)
    
    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        set_attr = {}
        for attr in self.special_tokens_attribute:
            attr_value = getattr(self, attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr
    
    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        set_attr = {}
        for attr in self.special_tokens_attribute:
            attr_value = self._special_tokens_map[attr]
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr
    
    @property
    def all_special_tokens(self) -> List[str]:
        all_tokens = []
        visited = set()
        for value in self.special_tokens_map.values():
            if not isinstance(value, list):
                tokens_to_add = [value] if value not in visited else []
            else:
                tokens_to_add = [val for val in value if val not in visited]
            visited.update(tokens_to_add)
            all_tokens.extend(tokens_to_add)
        return all_tokens
    
    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        all_tokens = []
        visited = set()
        for value in self.special_tokens_map.values():
            if not isinstance(value, list):
                tokens_to_add = [value] if str(value) not in visited else []
            else:
                tokens_to_add = [val for val in value if str(val) not in visited]
            visited.update(map(str, tokens_to_add))
            all_tokens.extend(tokens_to_add)
        return all_tokens

    @property
    def all_special_ids(self) -> List[int]:
        all_tokens = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_tokens)
        return all_ids
    
    def convert_tokens_to_ids(self, tokens: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        raise NotImplementedError

    def _add_model_specific_special_tokens(self, special_tokens):
        self.special_tokens_attribute.update(set(special_tokens.keys()))
        for key, value in special_tokens.items():
            if isinstance(value, (str, AddedToken)):
                self.special_tokens_map[key] = value
            else:
                raise TypeError(f"Special token {key} has to be either str or AddedToken but got: {type(value)}")

class HuggingFaceTokenizer(HuggingFaceSpecialToken, BaseTokenizer):
    # load from Tokenizer (fast version)
    
    vocab_files_names: Dict[str, str] = {"tokenizer_file": TOKENIZER_FILE}
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side: str = "right"
    truncation_side: str = "right"
    
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        tokenizer_object = kwargs.pop("tokenizer_object", None)
        tokenizer_file = kwargs.pop("tokenizer_file", None)
        added_tokens_decoder = kwargs.pop("added_tokens_decoder", {})
        self.add_prefix_space = kwargs.get("add_prefix_space", False)
        
        if tokenizer_object is not None:
            tokenizer = copy.deepcopy(tokenizer_object)
        elif tokenizer_file is not None:
            tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            raise ValueError("Couldn't instantiate the backend tokenizer")
        self._tokenizer = tokenizer
        
        _truncation = self._tokenizer.truncation
        if _truncation is not None:
            self._tokenizer.enable_truncation(**_truncation)
            kwargs.setdefault("max_length", _truncation["max_length"])
            kwargs.setdefault("truncation_side", _truncation["direction"])
            kwargs.setdefault("stride", _truncation["stride"])
            kwargs.setdefault("truncation_strategy", _truncation["strategy"])
        else:
            self._tokenizer.no_truncation()
        
        _padding = self._tokenizer.padding
        if _padding is not None:
            self._tokenizer.enable_padding(**_padding)
            kwargs.setdefault("pad_token", _padding["pad_token"])
            kwargs.setdefault("pad_token_type_id", _padding["pad_type_id"])
            kwargs.setdefault("padding_side", _padding["direction"])
            kwargs.setdefault("max_length", _padding["length"])
            kwargs.setdefault("pad_to_multiple_of", _padding["pad_to_multiple_of"])
        else:
            self._tokenizer.no_padding()

        for key in kwargs:
            if hasattr(self, key) and callable(getattr(self, key)):
                raise AttributeError(f"{key} conflicts with the method {key} in {self.__class__.__name__}")
        self.init_kwargs = copy.deepcopy(kwargs)
        self.name_or_path = kwargs.pop("name_or_path", "")
        model_max_length = kwargs.pop("model_max_length", kwargs.pop("max_len", None))
        self.model_max_length = model_max_length if model_max_length is not None else INIT_MODEL_MAX_LEN
        
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        if self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )
        
        self.truncation_side = kwargs.pop("truncation_side", self.truncation_side)
        if self.truncation_side not in ["right", "left"]:
            raise ValueError(
                f"Truncation side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )
        
        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)
        self.clean_up_tokenization_spaces = kwargs.pop("clean_up_tokenization_spaces", False)
        self.split_special_tokens = kwargs.pop("split_special_tokens", False)
        
        # Stores a Jinja template that formats chat histories into tokenizable strings
        self.prompt_template = kwargs.pop("prompt_template", None)
        if isinstance(self.prompt_template, (list, tuple)):
            # Chat templates are stored as lists of dicts with fixed key names,
            # we reconstruct that into a single dict while loading them.
            self.prompt_template = {template["name"]: template["template"] for template in self.prompt_template}
        
        self.extra_special_tokens = kwargs.pop("extra_special_tokens", {})
        self._add_model_specific_special_tokens(special_tokens=self.extra_special_tokens)

        # add special tokens
        self._tokenizer.encode_special_tokens = self.split_special_tokens
        added_tokens_decoder_hash = {hash(repr(token)) for token in self.added_tokens_decoder}
        tokens_to_add = [
            token
            for index, token in sorted(added_tokens_decoder.items(), key=lambda x: x[0])
            if hash(repr(token)) not in added_tokens_decoder_hash
        ]
        encoder = list(self.added_tokens_encoder.keys()) + [str(token) for token in tokens_to_add]
        # if some of the special tokens are strings, we check if we don't already have a token
        tokens_to_add += [
            token for token in self.all_special_tokens_extended if token not in encoder and token not in tokens_to_add
        ]

        if len(tokens_to_add) > 0:
            tokens = []
            special_tokens = self.all_special_tokens
            for token in tokens_to_add:
                is_special = (
                    (token.special or str(token) in special_tokens)
                    if isinstance(token, AddedToken)
                    else str(token) in special_tokens
                )
                if isinstance(token, str):
                    token = AddedToken(token, special=is_special)
                else:
                    token.special = is_special
                tokens.append(token)
            if tokens:
                self.add_tokens(tokens)

        if self._tokenizer.pre_tokenizer is not None:
            pre_tok_state = json.loads(self._tokenizer.pre_tokenizer.__getstate__())
            if pre_tok_state.get("add_prefix_space", self.add_prefix_space) != self.add_prefix_space:
                pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
                pre_tok_state["add_prefix_space"] = self.add_prefix_space
                self._tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=False)
    
    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()
    
    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        return self._tokenizer.get_added_tokens_decoder()

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        return {token.content: id for id, token in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}
    
    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> Tokenizer:
        return self._tokenizer
    
    @property
    def decoder(self) -> Decoder:
        return self._tokenizer.decoder
    
    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (`str`): The text to clean up.

        Returns:
            `str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    @classmethod
    def convert_added_tokens(cls, obj: Union[AddedToken, Any], save=False, add_type_field=True):
        if isinstance(obj, dict) and "__type" in obj and obj["__type"] == "AddedToken":
            obj.pop("__type")
            return AddedToken(**obj)
        if isinstance(obj, AddedToken) and save:
            obj = obj.__getstate__()
            if add_type_field:
                obj["__type"] = "AddedToken"
            else:
                # Don't save "special" for previous tokenizers
                obj.pop("special")
            return obj
        elif isinstance(obj, (list, tuple)):
            return [cls.convert_added_tokens(o, save=save, add_type_field=add_type_field) for o in obj]
        elif isinstance(obj, dict):
            return {k: cls.convert_added_tokens(v, save=save, add_type_field=add_type_field) for k, v in obj.items()}
        return obj

    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)
    
    def get_prompt_template(
        self,
        prompt_template: Optional[str] = None,
    ) -> Optional[str]:
        if isinstance(self.prompt_template, dict):
            template_dict = self.prompt_template
            if prompt_template is not None and prompt_template in template_dict:
                # The user can pass the name of a template to the chat template argument instead of an entire template
                prompt_template = template_dict[prompt_template]
            elif prompt_template is None:
                if "default" in template_dict:
                    prompt_template = template_dict["default"]
                else:
                    raise ValueError(
                        "This model has multiple chat templates with no default specified! Please either pass a chat "
                        "template or the name of the template you wish to use to the `prompt_template` argument. Available "
                        f"template names are {sorted(template_dict.keys())}."
                    )
        elif prompt_template is None:
            # These are the cases when the model has a single template
            # priority: `prompt_template` argument > `tokenizer.prompt_template`
            if self.prompt_template is not None:
                prompt_template = self.prompt_template

        return prompt_template
    
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
        train_on_all_assistant = kwargs.get("train_on_all_assistant", True)
        if return_dict and not tokenize:
            raise ValueError(
                "`return_dict=True` is incompatible with `tokenize=False`, because there is no dict "
                "of tokenizer outputs to return."
            )

        if return_mask and not return_dict:
            raise ValueError("`return_mask=True` is incompatible with `return_dict=False`")

        prompt_template = self.get_prompt_template(prompt_template)
        if prompt_template is not None:
            prompt_template = PromptTemplate(prompt_template)
            assistant_indices = None
            if return_mask:
                rendered_output, assistant_indices = prompt_template(
                    messages,
                    return_mask=return_mask,
                    train_on_all_assistant=train_on_all_assistant,
                    add_generation_prompt=add_generation_prompt
                )
            else:
                rendered_output = prompt_template(
                    messages,
                    return_mask=return_mask,
                    train_on_all_assistant=train_on_all_assistant,
                    add_generation_prompt=add_generation_prompt
                )
            if tokenize:
                out_encoding = self._batch_encode(rendered_output)[0]
                out = self._from_encoding_to_token_dict(out_encoding)
                if return_dict:
                    if return_mask:
                        current_mask = [1] * len(out["input_ids"])
                        for start_char_idx, end_char_idx in assistant_indices:
                            start_token_idx = out_encoding.char_to_token(start_char_idx)
                            end_token_idx = out_encoding.char_to_token(end_char_idx - 1)
                            for token_id in range(start_token_idx, end_token_idx + 1 if end_token_idx else len(out["input_ids"])):
                                current_mask[token_id] = 0
                        out["label_mask"] = current_mask
                    return out
                else:
                    return out["input_ids"]
            else:
                return rendered_output
        else:
            rendered_output = ""
            current_mask = []
            for message in messages:
                rendered_output += message["content"]
                if (train_on_all_assistant and message["role"] == "assistant") or message["masked"] == False:
                    current_mask.extend([0] * len(message["content"]))
                else:
                    current_mask.extend([1] * len(message["content"]))
            if tokenize:
                out = self.batch_encode(rendered_output)
                if return_dict:
                    out["label_mask"] = current_mask
                    return out
                else:
                    return out["input_ids"]
            else:
                return rendered_output
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        subfolder = kwargs.pop("subfolder", "")
        
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        vocab_files = {}
        init_configuration = {}
        is_local = os.path.isdir(pretrained_model_name_or_path)
        
        additional_files_names = {
            "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
            "tokenizer_file": TOKENIZER_FILE,
            "prompt_template_file": PROMPT_TEMPLATE_FILE,
        }
        vocab_files = {**cls.vocab_files_names, **additional_files_names}
        # Get file path (TODO: from url or cache)
        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            if file_path is None:
                resolved_vocab_files[file_id] = None
            if is_local:
                full_file_path = os.path.join(pretrained_model_name_or_path, subfolder, file_path)
                if os.path.exists(full_file_path):
                    resolved_vocab_files[file_id] = full_file_path
                else:
                    resolved_vocab_files[file_id] = None
                    logging.warning(f"{file_id} not found in {full_file_path}")
            else:
                # TODO: download from url
                pass
        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            cache_dir=cache_dir,
            **kwargs,
        )
    
    @classmethod
    def _from_pretrained(
        cls,
        resolved_vocab_files: Dict[str, str],
        pretrained_model_name_or_path: str,
        init_configuration,
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        tokenizer_file = resolved_vocab_files.pop("tokenizer_file", None)
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        
        assert tokenizer_file is not None and tokenizer_config_file is not None, \
            "Tokenizer file and tokenizer config file are not found."
        
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            init_kwargs = init_configuration
        
        # If an independent chat template file exists, it takes priority over template entries in the tokenizer config
        prompt_template_file = resolved_vocab_files.pop("prompt_template_file", None)
        if prompt_template_file is not None:
            with open(prompt_template_file) as prompt_template_handle:
                init_kwargs["prompt_template"] = prompt_template_handle.read()
        
        # Update with newly provided kwargs
        init_kwargs.update(kwargs)
        
        # Merge resolved_vocab_files arguments in init_kwargs.
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
        
        init_kwargs["name_or_path"] = pretrained_model_name_or_path
        init_kwargs["tokenizer_file"] = tokenizer_file
        
        # Handle tokenizer serialization of added and special tokens
        added_tokens_decoder: Dict[int, AddedToken] = {}
        added_tokens_map: Dict[str, AddedToken] = {}
        
        if "added_tokens_decoder" in init_kwargs:
            for idx, token in init_kwargs["added_tokens_decoder"].items():
                if isinstance(token, dict):
                    token = AddedToken(**token)
                if isinstance(token, AddedToken):
                    added_tokens_decoder[int(idx)] = token
                    added_tokens_map[str(token)] = token
                else:
                    raise ValueError(
                        f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance"
                    )
        else:
            with open(tokenizer_file, encoding="utf-8") as tokenizer_file_handle:
                tokenizer_file_handle = json.load(tokenizer_file_handle)
                added_tokens = tokenizer_file_handle.pop("added_tokens")
            for serialized_tokens in added_tokens:
                idx = serialized_tokens.pop("id")
                added_tokens_decoder[idx] = AddedToken(**serialized_tokens)
                added_tokens_map[str(added_tokens_decoder[idx])] = added_tokens_decoder[idx]
        
        init_kwargs["added_tokens_decoder"] = added_tokens_decoder
        init_kwargs = cls.convert_added_tokens(init_kwargs, save=False)
        for key in cls.special_tokens_attribute & init_kwargs.keys():
            if added_tokens_map != {} and init_kwargs[key] is not None:
                if key != "additional_special_tokens":
                    init_kwargs[key] = added_tokens_map.get(str(init_kwargs[key]), init_kwargs[key])
        
        tokenizer = cls(*init_inputs, **init_kwargs)
        return tokenizer

    def get_padding_and_truncation_strategy(
        self,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        **kwargs,
    ):
        if max_length is not None and padding is False and truncation is None:
            logging.warning("Truncation is not explicitly set but "
                            "'max_length' is provided. Defaulting to 'longest' truncation strategy.")
            truncation = "longest_first"
        
        # Get padding strategy
        if padding is not False:
            if padding is True:
                if max_length is not None:
                    if truncation is None or truncation is False or truncation == "no_truncate":
                        logging.warning("'max_length' is ignored when `padding`=`True` and there is no truncation. "
                                        "The padding strategy is set to `longest`. "
                                        "Please use `padding`='max_length' if you want to pad to max length.")
                        padding_strategy = PaddingStrategy.LONGEST
                    else:
                        padding_strategy = PaddingStrategy.MAX_LENGTH
                else:
                    padding_strategy = PaddingStrategy.LONGEST
            elif isinstance(padding, str):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.NO_PAD
        
        # Get truncation strategy
        if truncation is not False and truncation is not None:
            if truncation is True:
                truncation_strategy = TruncationStrategy.MAX_LENGTH
            elif isinstance(truncation, str):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.NO_TRUNCATE
        
        # Set max length
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length == INIT_MODEL_MAX_LEN:
                    logging.warning("`model_max_length` is not set, default to no padding.")
                    padding_strategy = PaddingStrategy.NO_PAD
                else:
                    max_length = self.model_max_length
            
            if truncation_strategy != TruncationStrategy.NO_TRUNCATE:
                if self.model_max_length == INIT_MODEL_MAX_LEN:
                    logging.warning("`model_max_length` is not set, default to no truncation.")
                    truncation_strategy = TruncationStrategy.NO_TRUNCATE
                else:
                    max_length = self.model_max_length
        
        # Check pad token
        if padding_strategy != PaddingStrategy.NO_PAD and (self.pad_token is None or self.pad_token_id < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )
        
        # Check we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.NO_TRUNCATE and
            padding_strategy != PaddingStrategy.NO_PAD and
            pad_to_multiple_of is not None and
            max_length is not None and
            (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                f"Truncation and padding are both activated but "
                f"truncation length {max_length} is not a multiple of pad_to_multiple_of {pad_to_multiple_of}."
            )
        
        return padding_strategy, truncation_strategy, max_length

    def set_padding_and_truncation_strategy(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
    ) -> Tuple[PaddingStrategy, TruncationStrategy]:
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding
        
        # Set padding strategy
        if padding_strategy == PaddingStrategy.NO_PAD:
            if _padding is not None:
                self._tokenizer.no_padding()
        else:
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {
                "length": length,
                "direction": padding_side if padding_side is not None else self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_type_id": self.pad_token_type_id,
                "pad_to_multiple_of": pad_to_multiple_of,
            }
            if _padding != target:
                self._tokenizer.enable_padding(**target)
        
        # Set truncation strategy
        if truncation_strategy == TruncationStrategy.NO_TRUNCATE:
            if _truncation is not None:
                self._tokenizer.no_truncation()
        else:
            target = {
                "max_length": max_length,
                "direction": self.truncation_side,
                "strategy": truncation_strategy.value,
                "stride": 0,
            }
            
            # _truncation may contain more keys than target
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}
            
            if current != target:
                self._tokenizer.enable_truncation(**target)
        
        return _padding, _truncation

    def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
        tokens_dict = self.batch_encode(text, **kwargs)
        return tokens_dict["input_ids"]

    def batch_encode(
        self,
        text: Union[str, List[str]],
        **kwargs: Dict[str, Any]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        verbose = kwargs.get("verbose", False)
        add_special_tokens = kwargs.get("add_special_tokens", True)
        max_length = kwargs.get("max_length", None)
        split_special_tokens = kwargs.get("split_special_tokens", self.split_special_tokens)
        
        if isinstance(text, str):
            batched_input = [text]
        else:
            batched_input = text
        if self._tokenizer.encode_special_tokens != split_special_tokens:
            self._tokenizer.encode_special_tokens = split_special_tokens
        
        encodings = self._batch_encode(
            batched_input,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        
        tokens_dict_list = []
        for encoding in encodings:
            tokens_dict = self._from_encoding_to_token_dict(encoding)
            tokens_dict_list.append(tokens_dict)
            
            # warning if some input_ids are too long
            if verbose and max_length is None and len(encoding.ids) > self.model_max_length:
                logging.warning(
                    f"Token indices sequence length is longer than the specified maximum sequence length for this model "
                    f"({len(encoding.ids)} > {self.model_max_length}). Running this sequence through the model will result in "
                    f"indexing errors"
                )
        
        if isinstance(text, str):
            return tokens_dict_list[0]
        return tokens_dict_list

    def _from_encoding_to_token_dict(
        self,
        encoding: Encoding,
    ) -> Dict[str, Union[List[int], List[str]]]:
        return_token_type_ids = "token_type_ids" in self.model_input_names
        return_attention_mask = "attention_mask" in self.model_input_names
        
        tokens_dict = {}
        tokens_dict["input_ids"] = encoding.ids
        if return_token_type_ids:
            tokens_dict["token_type_ids"] = encoding.type_ids
        if return_attention_mask:
            tokens_dict["attention_mask"] = encoding.attention_mask
        
        return tokens_dict

    def _batch_encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        **kwargs,
    ) -> List[Encoding]:
        if isinstance(text, str):
            text = [text]
        padding = kwargs.get("padding", False)
        truncation = kwargs.get("truncation", None)
        max_length = kwargs.get("max_length", None)
        pad_to_multiple_of = kwargs.get("pad_to_multiple_of", None)
        padding_side = kwargs.get("padding_side", self.padding_side)
        
        padding_strategy, truncation_strategy, max_length = self.get_padding_and_truncation_strategy(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        
        old_padding, old_truncation = self.set_padding_and_truncation_strategy(
            padding_strategy, truncation_strategy, max_length, pad_to_multiple_of, padding_side
        )
        
        encodings = self._tokenizer.encode_batch(
            text,
            add_special_tokens=add_special_tokens,
        )
        
        if old_padding is not None:
            self._tokenizer.enable_padding(**old_padding)
        if old_truncation is not None:
            self._tokenizer.enable_truncation(**old_truncation)
        
        return encodings
        
    def tokenize(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[str]:
        return self._encode(text, add_special_tokens=add_special_tokens, **kwargs).tokens
    
    def _decode(
        self,
        token_ids: Union[int, List[int]],
        **kwargs: Dict[str, Any],
    ) -> str:
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        clean_up_tokenization_spaces = kwargs.get("clean_up_tokenization_spaces", None)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if clean_up_tokenization_spaces is None:
            clean_up_tokenization_spaces = self.clean_up_tokenization_spaces
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)
        return text
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, id: int) -> str:
        return self._tokenizer.id_to_token(id)
    
    def convert_tokens_to_ids(self, tokens: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_unk(tokens)
        return [self._convert_token_to_id_with_unk(token) for token in tokens]        
            
    def _convert_token_to_id_with_unk(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_token_to_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        raise NotImplementedError
