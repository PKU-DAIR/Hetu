import os
import hydra
import hetu
from omegaconf import OmegaConf
from hetu.engine.sft_trainer import SFTTrainer
from hetu.engine.sft_config import SFTConfig
from hetu.models.llama import LlamaLMHeadModel
from hetu.models.llama import LlamaTokenizer
from hetu.utils.parallel.distributed import distributed_init
from hetu.data.messages.message_template import AlpacaTemplate
from hetu.data.messages.prompt_template import CHATML_TEMPLATE
from hetu.utils.parallel import read_ds_parallel_config

@hydra.main(config_path='conf', config_name='pretrain_pad', version_base=None)
def main(config):
    OmegaConf.resolve(config)
    distributed_init(config.rpc.num_gpus, config.rpc.server_addr, str(config.rpc.server_port))
    sft_config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(SFTConfig), config.trainer)
    )
    
    ds_parallel_config_path = sft_config.ds_parallel.ds_parallel_config_path
    ds_parallel_config_name = sft_config.ds_parallel.ds_parallel_config_name
    ds_parallel_config = os.path.join(ds_parallel_config_path, ds_parallel_config_name)
    if ds_parallel_config is None:
        raise ValueError("ds_parallel_config is required.")
    ds_parallel_configs = read_ds_parallel_config(ds_parallel_config)
    
    model = LlamaLMHeadModel.from_pretrained(
        config.model.pretrained_model_name_or_path,
        ds_parallel_configs,
    )
    tokenizer = LlamaTokenizer.from_pretrained(config.model.tokenizer.pretrained_model_name_or_path)
    trainer = SFTTrainer(
        sft_config,
        model,
        tokenizer,
        config.model.optimizer,
        message_template=AlpacaTemplate(),
        prompt_template=CHATML_TEMPLATE,
    )
    trainer.build()
    trainer.train()

if __name__ == "__main__":
    main()