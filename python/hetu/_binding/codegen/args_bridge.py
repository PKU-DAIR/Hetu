import os, sys, traceback
import re
from collections import OrderedDict

class ArgType:
    BOOL = 0; BOOL_STR = ("bool",)
    INT64 = 1; INT64_STR = ("int", "int64_t", "int64")
    FLOAT64 = 2; FLOAT64_STR = ("float", "double", "float64")
    STRING = 3; STRING_STR = ("str", "std::string", "string", "OpName", "TensorName")
    BOOL_LIST = 4; BOOL_LIST_STR = ("List[bool]", "BoolList", "std::vector<bool>", "vector<bool>", "HTKeepDims")
    INT64_LIST = 5; INT64_LIST_STR = ("List[int]", "IntList", "std::vector<int64_t>", "vector<int64_t>", "HTShape", "HTStride", "HTAxes")
    FLOAT64_LIST = 6; FLOAT64_LIST_STR = ("List[float]", "FloatList", "std::vector<double>", "vector<double>")
    STRING_LIST = 7; STRING_LIST_STR = ("List[string]", "List[str]", "StringList", "std::vector<std::string>", "vector<string>")
    PY_ARRAY = 8; PY_ARRAY_STR = ("numpy.array", "numpy.ndarray", "PyArray", "NumpyArray")
    PY_OBJECT = 9; PY_OBJECT_STR = ("PyObject*", "py_object", "py_obj")
    DATA_TYPE = 10; DATA_TYPE_STR = ("hetu.dtype", "dtype", "DataType")
    DEVICE = 11; DEVICE_STR = ("hetu.device", "device", "Device")
    DEVICE_GROUP = 12; DEVICE_GROUP_STR = ("hetu.DeviceGroup", "DeviceGroup")
    DG_HIERARCHY = 13; DG_HIERARCHY_STR = ("hetu.DeviceGroupHierarchy", "DeviceGroupHierarchy", "List[hetu.DeviceGroupList]", "List[DeviceGroupList]", "List[List[hetu.DeviceGroup]]", "List[List[DeviceGroup]]")
    STREAM = 14; STREAM_STR = ("hetu.stream", "stream", "Stream")
    ND_ARRAY = 15; ND_ARRAY_STR = ("hetu.NDArray", "NDArray", "hetu.ndarray", "ndarray")
    ND_ARRAY_LIST = 16; ND_ARRAY_LIST_STR = ("List[hetu.NDArray]", "List[NDArray]", "List[hetu.ndarray]", "List[ndarray]", "NDArrayList")
    TENSOR = 17; TENSOR_STR = ("hetu.Tensor", "Tensor", "hetu.tensor", "tensor")
    TENSOR_LIST = 18; TENSOR_LIST_STR = ("List[hetu.Tensor]", "List[Tensor]", "List[hetu.tensor]", "List[tensor]", "TensorList")
    OPERATOR = 19; OPERATOR_STR = ("hetu.Operator", "Operator", "hetu.operator", "operator")
    OPERATOR_LIST = 20; OPERATOR_LIST_STR = ("List[hetu.Operator]", "List[Operator]", "List[hetu.operator]", "List[operator]", "OperatorList", "OpList")
    FEED_DICT = 21; FEED_DICT_STR = ("FeedDict", "feed_dict")
    DISTRIBUTED_STATES = 22; DISTRIBUTED_STATES_STR = ("hetu.DistributedStates", "DistributedStates")
    DISTRIBUTED_STATES_LIST = 23; DISTRIBUTED_STATES_LIST_STR = ("List[hetu.DistributedStates]", "List[DistributedStates]", "DistributedStatesList") 
    DS_HIERARCHY = 24; DS_HIERARCHY_STR = ("List[hetu.DistributedStatesUnion]", "List[DistributedStatesUnion]", "DistributedStatesHierarchy") 
    INT_SYMBOL = 25; INT_SYMBOL_STR = ("hetu.IntSymbol", "IntSymbol")
    SYMBOLIC_SHAPE = 26; SYMBOLIC_SHAPE_STR = ("List[hetu.IntSymbol]", "List[IntSymbol]", "SyShape")
    SYMBOLIC_SHAPE_LIST = 27; SYMBOLIC_SHAPE_LIST_STR = ("List[List[[hetu.IntSymbol]]", "List[List[IntSymbol]]", "List[SyShape]")
    INITIALIZER = 28; INITIALIZER_STR = ("hetu.Initializer", "Initializer")
    SYMBOLIC_SHAPE_LIST_LIST = 29; SYMBOLIC_SHAPE_LIST_LIST_STR = ("List[List[List[[hetu.IntSymbol]]]", "List[List[List[IntSymbol]]]", "List[List[SyShape]]")

    

    # None is for returning type rather than argument type. 
    # We slightly abuse the notation here.
    NONE = -1; NONE_STR = ("None",)

    type_to_type_str_mapping = {}
    type_str_to_type_mapping = {}
    
    @classmethod
    def init_mapping(cls):
        if len(ArgType.type_to_type_str_mapping) != 0:
            return
        ArgType.type_to_type_str_mapping[ArgType.BOOL] = ArgType.BOOL_STR
        ArgType.type_to_type_str_mapping[ArgType.INT64] = ArgType.INT64_STR
        ArgType.type_to_type_str_mapping[ArgType.FLOAT64] = ArgType.FLOAT64_STR
        ArgType.type_to_type_str_mapping[ArgType.STRING] = ArgType.STRING_STR
        ArgType.type_to_type_str_mapping[ArgType.BOOL_LIST] = ArgType.BOOL_LIST_STR
        ArgType.type_to_type_str_mapping[ArgType.INT64_LIST] = ArgType.INT64_LIST_STR
        ArgType.type_to_type_str_mapping[ArgType.FLOAT64_LIST] = ArgType.FLOAT64_LIST_STR
        ArgType.type_to_type_str_mapping[ArgType.STRING_LIST] = ArgType.STRING_LIST_STR
        ArgType.type_to_type_str_mapping[ArgType.PY_ARRAY] = ArgType.PY_ARRAY_STR
        ArgType.type_to_type_str_mapping[ArgType.PY_OBJECT] = ArgType.PY_OBJECT_STR
        ArgType.type_to_type_str_mapping[ArgType.DATA_TYPE] = ArgType.DATA_TYPE_STR
        ArgType.type_to_type_str_mapping[ArgType.DEVICE] = ArgType.DEVICE_STR
        ArgType.type_to_type_str_mapping[ArgType.DEVICE_GROUP] = ArgType.DEVICE_GROUP_STR
        ArgType.type_to_type_str_mapping[ArgType.DG_HIERARCHY] = ArgType.DG_HIERARCHY_STR
        ArgType.type_to_type_str_mapping[ArgType.STREAM] = ArgType.STREAM_STR
        ArgType.type_to_type_str_mapping[ArgType.ND_ARRAY] = ArgType.ND_ARRAY_STR
        ArgType.type_to_type_str_mapping[ArgType.ND_ARRAY_LIST] = ArgType.ND_ARRAY_LIST_STR
        ArgType.type_to_type_str_mapping[ArgType.TENSOR] = ArgType.TENSOR_STR
        ArgType.type_to_type_str_mapping[ArgType.TENSOR_LIST] = ArgType.TENSOR_LIST_STR
        ArgType.type_to_type_str_mapping[ArgType.OPERATOR] = ArgType.OPERATOR_STR
        ArgType.type_to_type_str_mapping[ArgType.OPERATOR_LIST] = ArgType.OPERATOR_LIST_STR
        ArgType.type_to_type_str_mapping[ArgType.FEED_DICT] = ArgType.FEED_DICT_STR
        ArgType.type_to_type_str_mapping[ArgType.DISTRIBUTED_STATES] = ArgType.DISTRIBUTED_STATES_STR
        ArgType.type_to_type_str_mapping[ArgType.DISTRIBUTED_STATES_LIST] = ArgType.DISTRIBUTED_STATES_LIST_STR
        ArgType.type_to_type_str_mapping[ArgType.DS_HIERARCHY] = ArgType.DS_HIERARCHY_STR
        ArgType.type_to_type_str_mapping[ArgType.INT_SYMBOL] = ArgType.INT_SYMBOL_STR
        ArgType.type_to_type_str_mapping[ArgType.SYMBOLIC_SHAPE] = ArgType.SYMBOLIC_SHAPE_STR
        ArgType.type_to_type_str_mapping[ArgType.SYMBOLIC_SHAPE_LIST] = ArgType.SYMBOLIC_SHAPE_LIST_STR
        ArgType.type_to_type_str_mapping[ArgType.INITIALIZER] = ArgType.INITIALIZER_STR
        ArgType.type_to_type_str_mapping[ArgType.SYMBOLIC_SHAPE_LIST_LIST] = ArgType.SYMBOLIC_SHAPE_LIST_LIST_STR
        for t, ss in ArgType.type_to_type_str_mapping.items():
            for s in ss:
                ArgType.type_str_to_type_mapping[s] = t
    
    @classmethod
    def get_arg_type(cls, type_str):
        ret = ArgType.type_str_to_type_mapping.get(type_str)
        assert ret is not None, f"Argument type {type_str} does not exist"
        return ret
    
    @classmethod
    def get_ret_type(cls, type_str):
        if type_str in ArgType.NONE_STR:
            return ArgType.NONE
        ret = ArgType.type_str_to_type_mapping.get(type_str)
        assert ret is not None, f"Returning type {type_str} does not exist"
        return ret
    
    @classmethod
    def get_type_str(cls, arg_type):
        ret = ArgType.type_to_type_str_mapping.get(arg_type)
        assert ret is not None, f"Argument type {type_str} does not exist"
        return ret[0]

ArgType.init_mapping()

name_pattern = r"[a-zA-Z_][a-zA-Z0-9_]*"
type_name_pattern = r"(List\[NAME\])|(NAME)".replace("NAME", name_pattern)
positional_arg_pattern = r"TYPE NAME"\
    .replace("TYPE", type_name_pattern)\
    .replace("NAME", name_pattern)
keyword_arg_pattern = r"TYPE NAME=[^(, )]+"\
    .replace("TYPE", type_name_pattern)\
    .replace("NAME", name_pattern)
args_pattern = r"((ARG)*(, ARG)*(, KW_ARG)*)|((KW_ARG)*)"\
    .replace("KW_ARG", keyword_arg_pattern)\
    .replace("ARG", positional_arg_pattern)
args_pattern_re = re.compile(args_pattern)

def parse_args(args, kernel_or_operator, self_arg_name=None, ret_type_str=None):
    assert kernel_or_operator in ('kernel', 'operator')
    parsed_results = OrderedDict()

    if self_arg_name is not None:
        self_pos = -1
        args_excluding_self = []
    
    if ret_type_str is not None:
        assert kernel_or_operator == 'kernel'
        # plausible returning types
        plausible_ret_types = (
            ArgType.ND_ARRAY, ArgType.ND_ARRAY_LIST, 
            ArgType.NONE
        )
        # in which cases can we assign the "out" argument
        plausible_ret_types_with_out = (ArgType.ND_ARRAY,)

        ret_type = ArgType.get_ret_type(ret_type_str)
        assert ret_type in plausible_ret_types, \
            f"Invalid returning type {ret_type_str}"
        assert ret_type in plausible_ret_types_with_out, \
            f"Invalid args \"{args}\": " + \
            f"function returning {ret_type_str} " + \
            "should not assign the \"out\" argument"
        out_pos = -1

    assert args.strip() == args, f"Invalid args \"{args}\""
    match_obj = args_pattern_re.match(args)
    assert match_obj and match_obj.string == args, \
        f"Invalid args \"{args}\""
    
    arg_getters = []
    prev_has_default = False
    arg_pos = 0 # the position of for getters of parsed_args
    for arg in args.split(", "):
        assert arg.strip() == arg
        type_and_name = arg.split(" ")
        assert len(type_and_name) == 2, \
            f"Invalid args \"{args}\": argument \"{arg}\" is invalid"
        arg_type_str = type_and_name[0]
        arg_type = ArgType.get_arg_type(arg_type_str)
        has_default = type_and_name[1].find("=") != -1
        if has_default:
            name_and_default = type_and_name[1].split("=")
            assert len(name_and_default) == 2, \
                f"Invalid args \"{args}\": argument \"{arg}\" is invalid"
            name_str, default_str = name_and_default
            prev_has_default = True
        else:
            name_str = type_and_name[1]
            default_str = ""
            assert not prev_has_default, \
                f"Invalid args \"{args}\": " + \
                "non-optional arguments cannot follow optional arguments"
        
        if self_arg_name is not None:
            if name_str == self_arg_name:
                assert name_str != "out", "Name of self argument must not be \"out\""
                # do not increment the position for getters
                self_pos = arg_pos
                continue
            else:
                args_excluding_self.append(arg)

        if name_str == "out":
            assert ret_type_str is not None, \
                "The name \"out\" is reserved for outputs. " + \
                "Please use another name for the argument."
            assert arg_type == ret_type, \
                f"Invalid args \"{args}\": " + \
                f"argument \"{arg}\" does not match " + \
                f"returning type \"{ret_type_str}\"."
            assert has_default and default_str == "None", \
                f"Invalid args \"{args}\": " + \
                f"default value of \"{arg}\" must be None."
            out_pos = arg_pos
        
        getter_fn = get_arg_getter_fn(arg_type, default_str, has_default, 
                                      arg_type_str, args)
        arg_getters.append(f"parsed_args.{getter_fn}({arg_pos})")
        arg_pos += 1
    
    parsed_results['arg_getters'] = arg_getters
    
    if self_arg_name is not None:
        assert self_pos != -1, f"Invalid args \"{args}\": " + \
            f"argument \"{self_arg_name}\" does not exist for self."
        parsed_results['self_pos'] = self_pos
        parsed_results['args_excluding_self'] = ', '.join(args_excluding_self)
    
    if ret_type_str is not None:
        if ret_type in plausible_ret_types_with_out:
            assert out_pos >= 0, f"Invalid args \"{args}\": " + \
                f"argument \"{ret_type_str} out=None\" does not exist for the output."
        parsed_results['ret_type'] = ret_type
        parsed_results['out_pos'] = out_pos
    
    return parsed_results


def get_arg_getter_fn(arg_type, default_str, has_default, type_str, args):
    def assert_default_is_none(type_str, default_str):
        assert default_str == "None", \
            f"Invalid args \"{args}\": " + \
            f"default value of {type_str} can only be None " + \
            f"(got {default_str})"    
    
    if arg_type == ArgType.BOOL:
        if has_default:
            return "get_bool_or_default"
        else:
            return "get_bool"
    elif arg_type == ArgType.INT64:
        if has_default:
            return "get_int64_or_default"
        else:
            return "get_int64"
    elif arg_type == ArgType.FLOAT64:
        if has_default:
            return "get_float64_or_default"
        else:
            return "get_float64"
    elif arg_type == ArgType.STRING:
        if has_default:
            return "get_string_or_default"
        else:
            return "get_string"
    elif arg_type == ArgType.BOOL_LIST:
        if has_default:
            return "get_bool_list_or_default"
        else:
            return "get_bool_list"
    elif arg_type == ArgType.INT64_LIST:
        if has_default:
            return "get_int64_list_or_default"
        else:
            return "get_int64_list"
    elif arg_type == ArgType.FLOAT64_LIST:
        if has_default:
            return "get_float64_list_or_default"
        else:
            return "get_float64_list"
    elif arg_type == ArgType.STRING_LIST:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_string_list_optional"
        else:
            return "get_string_list"
    elif arg_type == ArgType.PY_ARRAY:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_numpy_array_optional"
        else:
            return "get_numpy_array"
    elif arg_type == ArgType.PY_OBJECT:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_py_obj_optional"
        else:
            return "get_py_obj"
    elif arg_type == ArgType.DATA_TYPE:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_dtype_or_peek"
        else:
            return "get_dtype"
    elif arg_type == ArgType.DEVICE:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_device_or_peek"
        else:
            return "get_device"
    elif arg_type == ArgType.DEVICE_GROUP:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_device_group_or_peek"
        else:
            return "get_device_group"
    elif arg_type == ArgType.DG_HIERARCHY:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_dg_hierarchy_or_peek"
        else:
            return "get_dg_hierarchy"
    elif arg_type == ArgType.STREAM:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_stream_or_peek"
        else:
            return "get_stream"
    elif arg_type == ArgType.ND_ARRAY:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_ndarray_optional"
        else:
            return "get_ndarray"
    elif arg_type == ArgType.ND_ARRAY_LIST:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_ndarray_list_or_empty"
        else:
            return "get_ndarray_list"
    elif arg_type == ArgType.TENSOR:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_tensor_optional"
        else:
            return "get_tensor"
    elif arg_type == ArgType.TENSOR_LIST:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_tensor_list_or_empty"
        else:
            return "get_tensor_list"
    elif arg_type == ArgType.OPERATOR:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_operator_optional"
        else:
            return "get_operator"
    elif arg_type == ArgType.OPERATOR_LIST:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_operator_list_or_empty"
        else:
            return "get_operator_list"
    elif arg_type == ArgType.FEED_DICT:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_feed_dict_or_empty"
        else:
            return "get_feed_dict"
    elif arg_type == ArgType.DISTRIBUTED_STATES:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_distributed_states_or_empty"
        else:
            return "get_distributed_states"
    elif arg_type == ArgType.DISTRIBUTED_STATES_LIST:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_distributed_states_list_or_empty"
        else:
            return "get_distributed_states_list"
    elif arg_type == ArgType.DS_HIERARCHY:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_ds_hierarchy_or_empty"
        else:
            return "get_ds_hierarchy"
    elif arg_type == ArgType.INT_SYMBOL:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_int_symbol_or_empty"
        else:
            return "get_int_symbol"
    elif arg_type == ArgType.SYMBOLIC_SHAPE:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_symbolic_shape_or_empty"
        else:
            return "get_symbolic_shape"
    elif arg_type == ArgType.SYMBOLIC_SHAPE_LIST:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_symbolic_shape_list_or_empty"
        else:
            return "get_symbolic_shape_list"
    elif arg_type == ArgType.INITIALIZER:
        return "get_initializer"
    elif arg_type == ArgType.SYMBOLIC_SHAPE_LIST_LIST:
        if has_default:
            assert_default_is_none(type_str, default_str)
            return "get_symbolic_shape_list_list_or_empty"
        else:
            return "get_symbolic_shape_list_list"
    else:
        raise Exception(
            f"Invalid args \"{args}\": type {type_str} is invalid")
