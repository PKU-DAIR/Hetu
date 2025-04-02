import numpy as np
from collections import UserDict

def is_pytorch(obj):
    try:
        import torch
        return isinstance(obj, torch.Tensor)
    except ImportError:
        return False

def is_tensorflow(obj):
    try:
        import tensorflow as tf
        return isinstance(obj, tf.Tensor)
    except ImportError:
        return False

def is_numpy(obj):
    return isinstance(obj, np.ndarray)

def is_hetu(obj):
    try:
        import hetu as ht
        return isinstance(obj, ht.Tensor)
    except ImportError:
        return False

def infer_framework(obj):
    preferred_framework = None
    representation = str(type(obj))
    if representation.startswith("<class 'numpy."):
        preferred_framework = "np"
    elif representation.startswith("<class 'hetu."):
        preferred_framework = "hetu"
    elif representation.startswith("<class 'tensorflow."):
        preferred_framework = "tf"
    elif representation.startswith("<class 'torch."):
        preferred_framework = "pt"
    
    if preferred_framework is None:
        if is_numpy(obj):
            return "np"
        elif is_hetu(obj):
            return "ht"
        elif is_pytorch(obj):
            return "pt"
        elif is_tensorflow(obj):
            return "tf"
        else:
            return None
    else:
        return preferred_framework

def to_py_obj(obj):
    framework_to_py_obj = {
        "pt": lambda obj: obj.detach().cpu().tolist(),
        "tf": lambda obj: obj.numpy().tolist(),
        "np": lambda obj: obj.tolist(),
        "ht": lambda obj: obj.numpy().tolist()
    }
    
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    
    inferred_framework = infer_framework(obj)
    if inferred_framework is None:
        return obj
    else:
        return framework_to_py_obj[inferred_framework](obj)
