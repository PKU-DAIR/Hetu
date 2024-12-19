import hetu
import re
from hetu.nn.modules.module import Module
from .layer import dispatch_lora_layer, dispatch_multi_lora_layers

class MultiLoraModel(Module):
    prefix: str = "lora"
    
    def __init__(self, model, peft_configs, config) -> None:
        super(MultiLoraModel, self).__init__()
        self.model = model
        self.targeted_module_names: dict[str, list[int]] = {}
        self.peft_configs = peft_configs
        self.config = config
        self.inject_adapter(self.model, self.config)
    
    def inject_adapter(self, model, config):
        key_list = [key for key, _ in model.named_modules()]
        
        for i, peft_config in enumerate(self.peft_configs):
            for key in key_list:
                if not self._check_target_module_exists(peft_config, key):
                    continue
                self.targeted_module_names.setdefault(key, []).append(i)
        
        for key, task_indices in self.targeted_module_names.items():
            parent, target, target_name = self._get_submodules(model, key)
            self._create_and_replace(self.peft_configs, config, target, target_name, parent, task_indices)
        
        self._mark_only_adapters_as_tranable(model)
    
    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        elif key in lora_config.target_modules:
            target_module_found = True
        else:
            target_module_found = any(key.endswith(f".{target_key}") for target_key in lora_config.target_modules)
        return target_module_found

    def _get_submodules(self, model, key):
        parent_key = ".".join(key.split(".")[:-1])
        parent = model.find_module(parent_key)
        target_name = key.split(".")[-1]
        target = model.find_module(key)
        return parent, target, target_name

    def _create_and_replace(self, lora_configs, config, target, target_name, parent, task_indices):
        ranks = []
        lora_alphas = []
        lora_dropouts = []
        use_rsloras = []
        for task_indice in task_indices:
            ranks.append(lora_configs[task_indice].rank)
            lora_alphas.append(lora_configs[task_indice].lora_alpha)
            lora_dropouts.append(lora_configs[task_indice].lora_dropout)
            use_rsloras.append(lora_configs[task_indice].use_rslora)
        kwargs = {
            "ranks": ranks,
            "lora_alphas": lora_alphas,
            "lora_dropouts": lora_dropouts,
            "use_rsloras": use_rsloras,
            "task_indices": task_indices
        }
        new_module = self._create_new_module(target, config, **kwargs)
        self._replace_module(parent, target_name, new_module, target)
    
    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias
    
    def _create_new_module(self, target, config, **kwargs):
        return dispatch_multi_lora_layers(target, config, **kwargs)

    def _mark_only_adapters_as_tranable(self, model) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.set_requires_grad(False)
    
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

class LoraModel(Module):
    prefix: str = "lora_"

    def __init__(self, model, config) -> None:
        super(LoraModel, self).__init__()
        self.model = model
        self.targeted_module_names: list[str] = []
        self.peft_config = config
        self.inject_adapter(self.model)
    
    def inject_adapter(self, model):
        key_list = [key for key, _ in model.named_modules()]
        
        for key in key_list:
            if not self._check_target_module_exists(self.peft_config, key):
                continue
            
            self.targeted_module_names.append(key)
            parent, target, target_name = self._get_submodules(model, key)
            self._create_and_replace(self.peft_config, target, target_name, parent)
        
        self._mark_only_adapters_as_tranable(model)
            
    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        elif key in lora_config.target_modules:
            target_module_found = True
        else:
            target_module_found = any(key.endswith(f".{target_key}") for target_key in lora_config.target_modules)
        return target_module_found

    def _get_submodules(self, model, key):
        parent_key = ".".join(key.split(".")[:-1])
        parent = model.find_module(parent_key)
        target_name = key.split(".")[-1]
        target = model.find_module(key)
        return parent, target, target_name

    def _create_and_replace(self, lora_config, target, target_name, parent):
        kwargs = {
            "rank": lora_config.rank,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "use_rslora": lora_config.use_rslora
        }
        
        new_module = self._create_new_module(target, **kwargs)
        self._replace_module(parent, target_name, new_module, target)
    
    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        new_module.base_layer._load_from_state_dict(
            child.state_dict(), hetu.local_device(), '', True, [], [], [])
    
    def _create_new_module(self, target, **kwargs):
        return dispatch_lora_layer(target, **kwargs)

    def _mark_only_adapters_as_tranable(self, model) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.set_requires_grad(False)
    
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
    
    