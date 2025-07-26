import hydra
from omegaconf import OmegaConf
from hetu.engine.trainer import Trainer
from hetu.engine.trainer_config import TrainingConfig
from hetu.engine.wrapper import ModelWrapperFromConfig
from hetu.utils.parallel.distributed import distributed_init

@hydra.main(config_path='conf', config_name='pretrain_pad', version_base=None)
def main(config):
    OmegaConf.resolve(config)
    distributed_init(config.rpc.num_gpus, config.rpc.server_addr, str(config.rpc.server_port))
    pretrain_config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TrainingConfig), config.trainer)
    )
    model = ModelWrapperFromConfig(config.model)
    trainer = Trainer(
        pretrain_config,
        model,
        config.model.tokenizer,
        config.model.optimizer,
    )
    trainer.build()
    trainer.train()

if __name__ == "__main__":
    main()