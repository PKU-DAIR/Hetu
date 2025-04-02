from typing import Optional
from dataclasses import dataclass, field
from hetu.engine.trainer_config import TrainingConfig
from hetu.peft.lora.config import LoraConfig
from hetu.data.messages.prompt_template import PromptTemplate

@dataclass
class SFTConfig(TrainingConfig):
    """
    Configuration class for Supervised Fine-Tuning (SFT).
    
    This class extends TrainingConfig to include SFT-specific settings such as
    PEFT configuration and prompt templates.
    """
    
    peft: Optional[LoraConfig] = field(
        default=None,
        metadata={"help": "PEFT configuration"},
    )

    prompt_template: Optional[str] = field(
        default=None,
        metadata={"help": "Prompt template for the prompt"},
    )

    def __post_init__(self):
        """
        Post-initialization processing to convert string prompt template to PromptTemplate object.
        """
        if self.prompt_template is not None:
            self.prompt_template = PromptTemplate(self.prompt_template)