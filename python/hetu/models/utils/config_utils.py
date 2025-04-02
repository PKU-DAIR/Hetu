import os
import copy
import json
import hetu
from typing import Any, Dict, Optional, Union, List
from hetu.models.utils.hub import is_remote_url
from hetu.models.utils import CONFIG_NAME

class PreTrainedConfig(object):
    """
    Base class for all model configurations.
    Stores model configuration attributes and provides utilities for loading/saving configurations.
    """
    model_type: str = ""
    sub_configs: Dict[str, "PreTrainedConfig"] = {}
    attribute_map: Dict[str, str] = {}
    
    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)
    
    def __init__(self, **kwargs):
        self.model_dtype = kwargs.pop("model_dtype", None)
        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        
        # Tokenizer arguments
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)
        
        if self.model_dtype is not None and isinstance(self.model_dtype, str):
            import hetu as ht
            self.model_dtype = getattr(ht, self.model_dtype)
    
    @property
    def name_or_path(self):
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        """
        Load configuration from a pretrained model configuration.
        
        Args:
            pretrained_model_name_or_path: Directory or name of the pretrained model.
            cache_dir: Cache directory for storing downloaded models.
            **kwargs: Additional arguments to update the configuration.
            
        Returns:
            PreTrainedConfig: Loaded configuration.
        """
        kwargs["cache_dir"] = cache_dir
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        **kwargs,        
    ):
        """
        Save configuration to a directory.
        
        Args:
            save_directory: Directory to save the configuration.
            **kwargs: Additional arguments.
            
        Raises:
            AssertionError: If save_directory is a file instead of a directory.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_dict(self) -> Dict[str, Any]:
        dict_without_symbol = self.__dict__.copy()
        pop_keys = []
        for key, value in dict_without_symbol.items():
            if (
                isinstance(value, hetu.IntSymbol) or
                "symbol" in key
            ):
                pop_keys.append(key)
        for key in pop_keys:
            dict_without_symbol.pop(key)
        print(f"dict_without_symbol: {dict_without_symbol}")
        output = copy.deepcopy(dict_without_symbol)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        del output["_name_or_path"]

        for key, value in output.items():
            if isinstance(value, PreTrainedConfig):
                value = value.to_dict()
            output[key] = value

        output['model_dtype'] = str(output['model_dtype']).split('.')[-1]
        return output

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ):
        """
        Get configuration dictionary from a pretrained model directory or file.
        
        Args:
            pretrained_model_name_or_path: Directory or file path to load from.
            **kwargs: Additional arguments.
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Configuration dictionary and updated kwargs.
            
        Raises:
            NotImplementedError: If remote URL handling is requested but not implemented.
        """
        subfolder = kwargs.pop("subfolder", "")
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder)):
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            # TODO: support remote download
            raise NotImplementedError
        else:
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME)
            resolved_config_file = os.path.join(pretrained_model_name_or_path, subfolder, configuration_file)
        
        if is_local:
            config_dict = cls._dict_from_json_file(resolved_config_file)
        else:
            # TODO: support remote download
            raise NotImplementedError
        return config_dict, kwargs
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        return cls(**config_dict)
    
    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)
    
    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def update(self, config_dict: Dict[str, Any]):
        """
        Update configuration with values from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters to update.
            
        Returns:
            PreTrainedConfig: Updated configuration instance.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self
