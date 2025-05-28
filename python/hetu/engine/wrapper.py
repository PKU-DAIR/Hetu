import hetu
from .utils import Args

class ModelWrapper(Args):
    """
    Wrapper class for model instantiation from model class and configuration.
    """
    def __init__(self, model_class, model_config):
        """
        Initialize a ModelWrapper.
        
        Args:
            model_class: Class of the model to instantiate.
            model_config: Configuration for the model.
        """
        self.model_class = model_class
        self.model_config = model_config

    def create_model(self, ds_parallel_configs):
        """
        Create a model instance with the provided parallel configurations.
        
        Args:
            ds_parallel_configs: DeepSpeed parallel configurations.
            
        Returns:
            Model instance created with the stored model class and configuration.
        """
        return self.model_class(config=self.model_config, ds_parallel_configs=ds_parallel_configs)

class ModelWrapperFromConfig(Args):
    """
    Wrapper class for model instantiation from a configuration object.
    """
    def __init__(self, config):
        """
        Initialize a ModelWrapperFromConfig.
        
        Args:
            config: Configuration object containing architecture and config_type.
        """
        self.model_class = getattr(hetu.models, config.architecture)
        self.model_config_class = getattr(hetu.models, config.config_type)
        self.model_config = self.model_config_class(**config)
    
    def create_model(self, ds_parallel_configs):
        """
        Create a model instance with the provided parallel configurations.
        
        Args:
            ds_parallel_configs: DeepSpeed parallel configurations.
            
        Returns:
            Model instance created with the stored model class and configuration.
        """
        return self.model_class(config=self.model_config, ds_parallel_configs=ds_parallel_configs)

class OptimizerWrapper(Args):
    def __init__(self, optimizer_config):
        optimizer_type = optimizer_config.pop('type')
        self.optimizer_class = getattr(hetu, optimizer_type + "Optimizer")
        self.optimizer_config = optimizer_config

    def create_optimizer(self, **kwargs):
        lr = self.optimizer_config.pop('learning_rate', None)
        sched = self.optimizer_config.pop('lr_scheduler', None)
        if sched is None:
            sched = {}
        sched.update({'lr': lr})
        sched.update(kwargs)
        return self.optimizer_class(**sched)
    
class DatasetWrapper(Args):
    def __init__(self, dataset_class):
        self.dataset_class = dataset_class
        
    def create_dataset(self, **kwargs):
        return self.dataset_class(**kwargs)
