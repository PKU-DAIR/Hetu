from typing import Optional, Callable, Union
from omegaconf import DictConfig
from torch.utils.data import Dataset
from hetu.engine.trainer import Trainer
from hetu.engine.sft_config import SFTConfig
from hetu.engine.wrapper import OptimizerWrapper
from hetu.models.utils.model_utils import PreTrainedModel
from hetu.data.tokenizers.pretrained_tokenizer import PreTrainedTokenizer
from hetu.data.messages.message_template import MessageTemplate
from hetu.data.dataset import JsonDataset, SFTDataset
from hetu.data.utils import convert_parquet_to_json

class SFTTrainer(Trainer):
    """
    Supervised Fine-Tuning Trainer for language models.
    
    This trainer extends the base Trainer class to provide specific functionality
    for supervised fine-tuning of language models, with support for PEFT methods
    and message templating.
    """
    
    def __init__(
        self,
        sft_config: SFTConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: Union[OptimizerWrapper, DictConfig],
        train_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        message_template: Optional[MessageTemplate] = None,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the SFT trainer.
        
        Args:
            sft_config (SFTConfig): Configuration for supervised fine-tuning
            model (PreTrainedModel): The model to fine-tune
            tokenizer (PreTrainedTokenizer): Tokenizer for text processing
            optimizer (Union[OptimizerWrapper, DictConfig]): Optimizer for training
            train_dataset (Optional[Dataset]): Dataset for training
            data_collator (Optional[Callable]): Function to collate batches
            message_template (Optional[MessageTemplate]): Template for formatting messages
            prompt_template (Optional[str]): Template for formatting prompts
            **kwargs: Additional keyword arguments
        """
        # peft
        # tokenizer - pad token = eos token if pad token is None
        self.sft_config = sft_config
        self.tokenizer = tokenizer
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.message_template = message_template
        self.prompt_template = prompt_template
        self.peft_config = self.sft_config.peft
        if self.peft_config is not None:
            from hetu.peft.lora import LoraModel
            model = LoraModel(model, self.peft_config) # TODO: peft PreTrainedModel
        # prepare dataset
        if train_dataset is None:
            train_dataset = self._prepare_dataset(
                tokenizer,
                sft_config.dataset_text_field,
                self.message_template,
                self.prompt_template,
            )

        super().__init__(
            sft_config,
            model,
            tokenizer,
            optimizer,
            train_dataset,
            data_collator,
        )
    
    def _prepare_dataset(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_text_field: Optional[str],
        message_template: Optional[MessageTemplate],
        prompt_template: Optional[str],
    ):
        """
        Prepare the training dataset based on config settings.
        
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer for processing texts
            dataset_text_field (Optional[str]): Field name for text in JSON dataset
            message_template (Optional[MessageTemplate]): Template for formatting messages
            prompt_template (Optional[str]): Template for formatting prompts
        
        Returns:
            Dataset: The prepared dataset for training
        """
        train_dataset_path = self.sft_config.train_dataset_path
        if train_dataset_path.endswith(".parquet"):
            train_dataset_path = convert_parquet_to_json(train_dataset_path)

        if message_template is None:
            return JsonDataset(
                train_dataset_path,
                dataset_text_field,
                tokenizer,
                self.sft_config.max_seq_length,
            )
        else:
            return SFTDataset(
                train_dataset_path,
                tokenizer,
                self.sft_config.max_seq_length,
                message_template,
                prompt_template,
            )