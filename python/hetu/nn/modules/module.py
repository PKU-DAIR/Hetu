import hetu
import itertools
from contextlib import ExitStack, nullcontext
from hetu import Tensor
from ..parameter import Parameter
from collections import OrderedDict, namedtuple

from typing import Union, Tuple, List, Dict, Set, Any, Callable, Iterator, Optional, TypeVar
from hetu import _tensor_type_t
from ..parameter import _param_type_t

_module_type_t = TypeVar('T', bound='Module')
_member_type_t = Union[_param_type_t, _module_type_t, _tensor_type_t]
_member_t = Union[Optional[Parameter], Optional['Module'], Optional[Tensor]]


def parallel_data_provider(global_data, ds, device_index):
    order, states = ds.order, ds.states
    local_map = hetu.map_to_local_data(ds, device_index)
    local_data = global_data.copy()
    dims = len(local_data.shape)
    begin_pos = [0] * dims
    out_shape = local_data.shape
    for dim in order:
        if dim < 0:
            continue
        splits = states[dim]
        split_index = local_map[dim]
        start = int(split_index * (global_data.shape[dim] / splits))
        stop = min(int((split_index + 1) * (global_data.shape[dim] / splits)), global_data.shape[dim])
        if isinstance(local_data, hetu.NDArray): 
            begin_pos[dim] = start
            out_shape[dim] = stop - start
        else:
            local_data = local_data.take(range(start, stop), axis=dim)
    if isinstance(local_data, hetu.NDArray):
        local_data = local_data.slice(begin_pos, out_shape)
    return local_data


class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


class Module(object):

    def __init__(self):
        self._recompute = False
        self._output_recompute = False
        self._cpu_offload = False
        self.module_name = None
        self.ds_parallel_configs = None
        self.global_name = ""
        with hetu.graph("define_and_run"):
            super().__setattr__("_parameters", OrderedDict())
            super().__setattr__("_modules", OrderedDict())
            super().__setattr__("_buffers", OrderedDict())
            super().__setattr__("_modules", OrderedDict())
            super().__setattr__("_non_persistent_buffers_set", set())
    
    def __getattr__(self, name: str) -> Any:
        with hetu.graph("define_and_run"):
            _parameters = self.__dict__.get('_parameters')
            if _parameters is not None:
                param = _parameters.get(name)
                if param is not None:
                    return param
            _buffers = self.__dict__.get('_buffers')
            if _buffers is not None:
                buffers = _buffers.get(name)
                if buffers is not None:
                    return buffers
            _modules = self.__dict__.get('_modules')
            if _modules is not None:
                module = _modules.get(name)
                if module is not None:
                    return module
            return None
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        with hetu.graph("define_and_run"):
            def remove_from_members(*members):
                for dict_or_set in members:
                    if dict_or_set is not None and name in dict_or_set:
                        if isinstance(dict_or_set, dict):
                            del dict_or_set[name]
                        else:
                            dict_or_set.discard(name)
            
            _parameters = self.__dict__.get('_parameters')
            _modules = self.__dict__.get('_modules')
            _buffers = self.__dict__.get('_buffers')
            _non_persistent_buffers_set = self.__dict__.get('_non_persistent_buffers_set')
            
            # Parameters
            if isinstance(value, hetu.Tensor):
                remove_from_members(self.__dict__, _modules, _buffers, 
                                    _non_persistent_buffers_set)
                self.register_parameter(name, value)
                return
            
            # Modules
            if isinstance(value, hetu.nn.Module):
                remove_from_members(self.__dict__, _parameters, _buffers, 
                                    _non_persistent_buffers_set)
                self.register_module(name, value)
                return
            
            # Buffers
            if isinstance(value, hetu.Tensor):
                remove_from_members(self.__dict__, _parameters, _modules)
                self.register_buffer(name, value, persistent=True)
                return
            
            remove_from_members(self.__dict__, _parameters, _modules, _buffers, 
                                _non_persistent_buffers_set)
            super().__setattr__(name, value)
    
    def __delattr__(self, name: str) -> None:
        with hetu.graph("define_and_run"):
            _parameters = self.__dict__.get('_parameters')
            if _parameters is not None and name in _parameters:
                del _parameters[name]
                return
            
            _modules = self.__dict__.get('_modules')
            if _modules is not None and name in _modules:
                del _modules[name]
                return
            
            _buffers = self.__dict__.get('_buffers')
            if _buffers is not None and name in _buffers:
                del _buffers[name]
                _non_persistent_buffers_set = self.__dict__.get('_non_persistent_buffers_set')
                del _non_persistent_buffers_set[name]
                return
            
            super().__delattr__(name)
        
    
    def __dir__(self):
        module_attrs = dir(self.__class__)
        self_attrs = list(self.__dict__.keys())
        _parameters = list(self._parameters.keys())
        _modules = list(self._modules.keys())
        _buffers = list(self._buffers.keys())
        keys = module_attrs + self_attrs + _parameters + _modules + _buffers
        return sorted(keys)
    
    ############################################################################
    # Registration of members
    ############################################################################
    
    def _register_member(self, name: str, value: _member_t, members: dict, 
                         reg_type: _member_type_t) -> None:
        with hetu.graph("define_and_run"):
            if not isinstance(name, str):
                raise TypeError(
                    f"Name of {reg_type.__name__} must be string "
                    f"(got {type(name).__name__})")
            if name == '':
                raise KeyError(f"Name of {reg_type.__name__} must not be empty")
            if '.' in name:
                raise KeyError(
                    f"Name of {reg_type.__name__} must not contain \".\" "
                    f"(got \"{name}\")")
            
            if value is None:
                members[name] = None
            elif not isinstance(value, reg_type):
                raise TypeError(
                    f"Cannot register a '{type(value).__name__}' object as "
                    f"a {reg_type.__name__} object")
            else:
                members[name] = value
    
    def register_parameter(self, name: str, value: Optional[Tensor]) -> None:
        with hetu.graph("define_and_run"):
            _parameters = self.__dict__.get('_parameters')
            if _parameters is None:
                raise AttributeError(
                    "Cannot register parameters before calling Module.__init__()")
            self._register_member(name, value, _parameters, Tensor)
    
    def register_module(self, name: str, value: Optional['Module']) -> None:
        with hetu.graph("define_and_run"):
            _modules = self.__dict__.get('_modules')
            if _modules is None:
                raise AttributeError(
                    "Cannot register modules before calling Module.__init__()")
            value.__setattr__("module_name", name)
            value.global_name = self.global_name + "." + name 
            self._register_member(name, value, _modules, Module)
    
    def add_module(self, name: str, value: Optional['Module']) -> None:
        with hetu.graph("define_and_run"):
            self.register_module(name, value)
    
    def register_buffer(self, name: str, value: Optional[Tensor], 
                        persistent: bool = True) -> None:
        with hetu.graph("define_and_run"):
            _buffers = self.__dict__.get('_buffers')
            if _buffers is None:
                raise AttributeError(
                    "Cannot register buffers before calling Module.__init__()")
            self._register_member(name, value, _buffers, Tensor)
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    ############################################################################
    # Iterator/Generator of members
    ############################################################################
    
    def _named_members(self, get_members_fn: Callable, prefix: str = '', 
                       recurse: bool = True) -> Iterator[Tuple[str, Any]]:
        visited = set()
        if recurse:
            modules = self.named_modules(prefix=prefix)
        else:
            modules = [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in visited:
                    continue
                visited.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        return self._named_members(
            lambda m: m._parameters.items(), 
            prefix=prefix, 
            recurse=recurse)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param
    
    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', 
                      remove_duplicate: bool = True) -> Iterator[Tuple[str, 'Module']]:
        memo = memo or set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is not None:
                    sub_prefix = prefix + ('.' if prefix else '') + name
                    for m in module.named_modules(memo, sub_prefix, remove_duplicate):
                        yield m
    
    def find_module(self, key: str = '') -> Optional['Module']:
        if key == '':
            return self
        if '.' not in key:
            for name, module in self._modules.items():
                if name == key:
                    return module
        key_list = key.split(".")
        for name, module in self._modules.items():
            if name == key_list[0]:
                sub_prefix = ".".join(key_list[1:])
                return module.find_module(sub_prefix)
    
    def modules(self) -> Iterator['Module']:
        for _, module in self.named_modules():
            yield module
    
    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        return self._named_members(
            lambda m: m._buffers.items(), 
            prefix=prefix, 
            recurse=recurse)
    
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer
    
    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        memo = set()
        for name, child in self._modules.items():
            if child is not None and child not in memo:
                memo.add(child)
                yield name, child

    def children(self) -> Iterator['Module']:
        for _, child in self.named_children():
            yield child
    
    ############################################################################
    # Call and Forward
    ############################################################################

    def __call__(self, *input, **kwargs) -> Any:
        with ExitStack() as stack:
            stack.enter_context(hetu.graph("define_and_run"))
            if self.module_name is None:
                print(self.__class__.__name__, "doesn't have module_name")
            else:
                # print(self.__class__.__name__,  "Module_Name:", self.module_name, "Global_name:", self.global_name)
                stack.enter_context(hetu.subgraph(name = self.module_name, module_type = self.__class__.__name__))
                _parameters = self.__dict__.get('_parameters')
                for key, param in _parameters.items():
                    # print(self.module_name, " ", key, " ", type(param), param)
                    hetu.add_to_subgraph(param)
            
            # multi-strategies of recompute  
            if self._recompute:
                mutli_recompute = [[True] for _ in range(1 if self.ds_parallel_configs is None else len(self.ds_parallel_configs))]  
            elif self.ds_parallel_configs is not None:
                mutli_recompute = [[False] if 'recompute' not in self.ds_parallel_configs[i] else self.ds_parallel_configs[i]['recompute'] for i in range(len(self.ds_parallel_configs))]  
            else:
                mutli_recompute = [[False]]  
                print(f"Warning: {self.__class__.__name__} does not have ds_parallel_configs attribute, undefined behavior may happen")
                
            # multi-strategies of cpu offload
            if self._cpu_offload:
                mutli_cpu_offload = [[True] for _ in range(1 if self.ds_parallel_configs is None else len(self.ds_parallel_configs))]
            elif self.ds_parallel_configs is not None:
                mutli_cpu_offload = [[False] if 'cpu_offload' not in self.ds_parallel_configs[i] else self.ds_parallel_configs[i]['cpu_offload'] for i in range(len(self.ds_parallel_configs))]
            else:
                mutli_cpu_offload = [[False]]
                print(f"Warning: {self.__class__.__name__} does not have ds_parallel_configs attribute, undefined behavior may happen")
                
            # for name, child in self.named_children():
            #     print("Sub:", name, ",", child)
            # print(self.module_name, self.ds_parallel_configs)
            stack.enter_context(hetu.recompute(mutli_recompute))
            stack.enter_context(hetu.cpu_offload(mutli_cpu_offload))

            # 实际forward
            value = self.forward(*input, **kwargs)
            
            # block之间显式地插入了comm op从而阻隔了连续的recompute
            # 之前需要在这里手动设置block的output不做recompute
            
            # 手动设置module的output不做recompute，从ds config传入
            if isinstance(value, hetu.Tensor):
                value._make_recompute(
                    [False] if self.ds_parallel_configs is None else
                    [ds_parallel_config.get('output_recompute', [False]) for ds_parallel_config in self.ds_parallel_configs]
                )
            return value
    
    def forward(self, *input: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Forward of module '{type(self).__name__}' is not implemented")
        
    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        _parameters = self.__dict__.get('_parameters')
        for key, param in _parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            # param_applied = fn(param)
            # should_use_set_data = compute_should_use_set_data(param, param_applied)
            # if should_use_set_data:
            #     param.data = param_applied
            #     out_param = param
            # else:
            #     assert isinstance(param, Parameter)
            #     assert param.is_leaf
            # out_param = Parameter(param_applied, param.requires_grad)
            out_param = fn(param)
            _parameters[key] = out_param

        _buffers = self.__dict__.get('_buffers')
        for key, buf in _buffers.items():
            if buf is not None:
                _buffers[key] = fn(buf)

        return self        
    
    def to(self, dtype=None, device=None):
        if dtype == None and device == None:
            return
        def convert(t):
            return t.to(dtype, device)
        return self._apply(convert) 
    
    def recompute(self, output_recompute=False):
        self._recompute = True
        self._output_recompute = output_recompute
        return self

    def cpu_offload(self):
        self._cpu_offload = True
        return self
    
    ############################################################################
    # Save and Load
    ############################################################################ 

    def _save_to_state_dict(self, destination, prefix, format='numpy'):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~hetu.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        _parameters = self.__dict__.get('_parameters')
        for name, param in _parameters.items():
            if param is not None:
                if format == 'numpy':
                    destination[prefix + name] = param.get_data()
                elif format == 'hetu':
                    destination[prefix + name] = param
                else:
                    raise NotImplementedError("state_dict() can only use numpy.ndarray or hetu.Tensor.")
        _buffers = self.__dict__.get('_buffers')
        for name, buf in _buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                if format == 'numpy':
                    destination[prefix + name] = buf.get_data()
                elif format == 'hetu': 
                    destination[prefix + name] = buf
                else:
                    raise NotImplementedError("state_dict() can only use numpy.ndarray or hetu.Tensor.")
                
    def state_dict(self, destination=None, prefix='', format='numpy'):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix, format)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', format)
        return destination
    
    def _load_from_state_dict(self, state_dict, local_device, prefix, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~hetu.nn.Module.load_state_dict`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~hetu.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                param_data = state_dict[key]
                # print("param_date_type", type(param_data))
                try:
                    if local_device is None:
                        # Tensor
                        assert param.shape == list(param_data.shape), "shape mismatched!"
                        param.reset_data(param_data)
                    else:
                        # Distributed Tensor
                        assert param.global_shape == list(param_data.shape), "global shape mismatched!"
                        device_group = param.get_device_group()
                        assert device_group.num_devices > 0, f"device group has {device_group.num_devices} devices, which is illegal, please check your model initialization."
                        # 3D parallel: for pipeline situation, a device don't need to load all the checkpoint
                        if device_group.contains(local_device):
                            device_index = device_group.get_index(local_device)
                            data = parallel_data_provider(param_data, param.distributed_states, device_index)
                            param.reset_data(data)
                            '''
                            if 'lm_head' in key:
                                print('lm_head', parallel_data_provider(param_data, param.distributed_states, device_index), 
                                        'param.get_data():', param.get_data())
                            '''
                except Exception as ex:
                    error_msgs.append('While resetting the parameter named "{}", '
                                      'whose global dimensions in the model are {} and '
                                      'whose global dimensions in the checkpoint are {}, '
                                      'an exception occurred : {}.'
                                      .format(key, param.global_shape, list(param_data.shape), ex.args))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)
                        
    def load_state_dict(self, state_dict, local_device = None, strict = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~hetu.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        state_dict = state_dict.copy()
    
        def load(module, prefix=''):
            module._load_from_state_dict(
                state_dict, local_device, prefix, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)