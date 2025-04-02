#include "hetu/_binding/graph/graph.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/eager_graph.h"
#include "hetu/graph/define_and_run_graph.h"

namespace hetu {
namespace graph {

PyObject* PyGraph_New(GraphId graph_id) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyGraph_Type->tp_alloc(PyGraph_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyGraph";
  auto* self = reinterpret_cast<PyGraph*>(unsafe_self);
  // new(&self->graph_id) GraphId();
  self->graph_id = graph_id;
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyGraph_dealloc(PyDevice* self) {
  // (&self->graph_id)->~GraphId();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyGraph_str(PyGraph* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(Graph::GetGraph(self->graph_id).name());
  HT_PY_FUNC_END
}

PyObject* PyGraph_repr(PyGraph* self) {
  return PyGraph_str(self);
}

PyObject* PyGraph_id(PyGraph* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->graph_id);
  HT_PY_FUNC_END
}

PyObject* PyGraph_name(PyGraph* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(Graph::GetGraph(self->graph_id).name());
  HT_PY_FUNC_END
}

PyObject* PyGraph_num_strategy(PyGraph* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(Graph::GetGraph(self->graph_id).NUM_STRATEGY);
  HT_PY_FUNC_END
}

PyObject* PyGraph_cur_strategy_id(PyGraph* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(Graph::GetGraph(self->graph_id).CUR_STRATEGY_ID);
  HT_PY_FUNC_END
}

PyObject* PyGraph_use_hetero_id(PyGraph* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(Graph::GetGraph(self->graph_id).USE_HETERO_ID);
  HT_PY_FUNC_END
}

PyObject* PyGraph_cur_hetero_id(PyGraph* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(Graph::GetGraph(self->graph_id).CUR_HETERO_ID);
  HT_PY_FUNC_END
}

PyObject* PyGraph_set_num_strategy(PyGraph* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"set_num_strategy(int num_strategy)"});
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    Graph::GetGraph(self->graph_id).NUM_STRATEGY = parsed_args.get_int64(0);
    return PyLong_FromInteger(Graph::GetGraph(self->graph_id).NUM_STRATEGY);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }  
  HT_PY_FUNC_END
}

PyObject* PyGraph_merge_strategy(PyGraph* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"merge_strategy(int graph_id)"});
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto& graph = Graph::GetGraph(self->graph_id);
    auto& another_graph = Graph::GetGraph(parsed_args.get_int64(0));
    HT_ASSERT(graph.type() == GraphType::DEFINE_AND_RUN && another_graph.type() == GraphType::DEFINE_AND_RUN)
      << "Currently only support merge two define graph";
    dynamic_cast<DefineAndRunGraph&>(graph).MergeGraph(dynamic_cast<DefineAndRunGraph&>(another_graph));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }  
  HT_PY_FUNC_END
}

PyObject* PyGraph_run(PyGraph* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "run(Tensor fetch, FeedDict feed_dict=None)", 
    "run(List[Tensor] fetches, FeedDict feed_dict=None)",
    "run(Tensor loss, Tensor fetch, FeedDict feed_dict=None, IntSymbolDict int_symbol_dict=None, int num_micro_batches=1)", 
    "run(Tensor loss, List[Tensor] fetches, FeedDict feed_dict=None, IntSymbolDict int_symbol_dict=None, int num_micro_batches=1, \
         int cur_strategy_id=0, int run_level=0, bool save_checkpoint=False, double grad_scale=1, int num_tokens=0)",
    "run(Tensor loss, List[Tensor] fetches, FeedDict feed_dict=None, IntSymbolDict int_symbol_dict=None, int num_micro_batches=1, \
         int compute_strategy_id=0, int optimize_strategy_id=0, int run_level=0, bool save_checkpoint=False, double grad_scale=1, int num_tokens=0)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyNDArrayList_New(Graph::GetGraph(self->graph_id).Run(
      {parsed_args.get_tensor(0)}, 
      parsed_args.get_feed_dict_or_empty(1)));
  } else if (parsed_args.signature_index() == 1) {
    return PyNDArrayList_New(Graph::GetGraph(self->graph_id).Run(
      parsed_args.get_tensor_list(0), 
      parsed_args.get_feed_dict_or_empty(1)));
  } else if (parsed_args.signature_index() == 2) {
    return PyNDArrayList_New(Graph::GetGraph(self->graph_id).Run(
      parsed_args.get_tensor(0),
      {parsed_args.get_tensor(1)}, 
      parsed_args.get_feed_dict_or_empty(2),
      parsed_args.get_int_symbol_dict_or_empty(3),
      parsed_args.get_int64_or_default(4)));
  } else if (parsed_args.signature_index() == 3) {
    auto run_level = static_cast<RunLevel>(parsed_args.get_int64_or_default(6));
    auto acc_num_tokens = Graph::GetGraph(self->graph_id).num_tokens();
    if (run_level == RunLevel::GRAD || run_level == RunLevel::UPDATE) {
      auto num_tokens = parsed_args.get_int64_or_default(9);
      acc_num_tokens += num_tokens;
      Graph::GetGraph(self->graph_id).set_num_tokens(acc_num_tokens);
    }
    auto* ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 1, PyLong_FromInteger(acc_num_tokens));
    PyObject* results = PyNDArrayList_New(Graph::GetGraph(self->graph_id).Run(
      parsed_args.get_tensor(0),
      parsed_args.get_tensor_list(1), 
      parsed_args.get_feed_dict_or_empty(2),
      parsed_args.get_int_symbol_dict_or_empty(3),
      parsed_args.get_int64_or_default(4),
      parsed_args.get_int64_or_default(5),
      parsed_args.get_int64_or_default(5),
      static_cast<RunLevel>(parsed_args.get_int64_or_default(6)),
      parsed_args.get_bool_or_default(7),
      parsed_args.get_float64_or_default(8)));
    if (run_level == RunLevel::UPDATE) {
      Graph::GetGraph(self->graph_id).set_num_tokens(0);
    }
    PyTuple_SET_ITEM(ret, 0, results);
    return ret;
  } else if (parsed_args.signature_index() == 4) {
    auto run_level = static_cast<RunLevel>(parsed_args.get_int64_or_default(7));
    auto acc_num_tokens = Graph::GetGraph(self->graph_id).num_tokens();
    if (run_level == RunLevel::GRAD || run_level == RunLevel::UPDATE) {
      auto num_tokens = parsed_args.get_int64_or_default(10);
      acc_num_tokens += num_tokens;
      Graph::GetGraph(self->graph_id).set_num_tokens(acc_num_tokens);
    }
    auto* ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 1, PyLong_FromInteger(acc_num_tokens));
    PyObject* results = PyNDArrayList_New(Graph::GetGraph(self->graph_id).Run(
      parsed_args.get_tensor(0),
      parsed_args.get_tensor_list(1), 
      parsed_args.get_feed_dict_or_empty(2),
      parsed_args.get_int_symbol_dict_or_empty(3),
      parsed_args.get_int64_or_default(4),
      parsed_args.get_int64_or_default(5),
      parsed_args.get_int64_or_default(6),
      static_cast<RunLevel>(parsed_args.get_int64_or_default(7)),
      parsed_args.get_bool_or_default(8),
      parsed_args.get_float64_or_default(9)));
    if (run_level == RunLevel::UPDATE) {
      Graph::GetGraph(self->graph_id).set_num_tokens(0);
    }
    PyTuple_SET_ITEM(ret, 0, results);
    return ret;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyGraph_get_graph(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "get_graph(int graph_id)",
    "get_graph(str graph_name)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    // Call `Graph::GetGraph` to check whether `graph_id` is valid
    return PyGraph_New(Graph::GetGraph(parsed_args.get_int64(0)).id());
    Py_RETURN_NONE;
  } else if (parsed_args.signature_index() == 1) {
    return PyGraph_New(Graph::GetGraph(parsed_args.get_string(0)).id());
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyGraph_delete_graph(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "delete_graph(int graph_id)",
    "delete_graph(str graph_name)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    // Call `Graph::GetGraph` to check whether `graph_id` is valid
    Graph::DeleteGraph(parsed_args.get_int64(0));
    Py_RETURN_NONE;
  } else if (parsed_args.signature_index() == 1) {
    Graph::DeleteGraph(parsed_args.get_string(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyGraph_get_default_define_and_run_graph(PyObject*) {
  HT_PY_FUNC_BEGIN
  return PyGraph_New(Graph::get_default_define_and_run_graph().id());
  HT_PY_FUNC_END
}

PyObject* PyGraph_get_default_define_by_run_graph(PyObject*) {
  HT_PY_FUNC_BEGIN
  return PyGraph_New(Graph::get_default_define_by_run_graph().id());
  HT_PY_FUNC_END
}

PyObject* PyGraph_get_default_eager_graph(PyObject*) {
  HT_PY_FUNC_BEGIN
  return PyGraph_New(Graph::get_default_eager_graph().id());
  HT_PY_FUNC_END
}

PyObject* PyGraph_make_new_eager_graph(PyObject*, PyObject* args,
                                       PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "make_new_eager_graph(str name, int init_capacity=None)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyGraph_New(
      Graph::make_new_graph<EagerGraph>(
        parsed_args.get_string(0),
        parsed_args.get_int64_or_else(1, Graph::DEFAULT_GRAPH_INITIAL_CAPACITY))
        .id());
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyGraph_make_new_define_and_run_graph(PyObject*, PyObject* args,
                                       PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "make_new_define_and_run_graph(str name, int init_capacity=None)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyGraph_New(
      Graph::make_new_graph<DefineAndRunGraph>(
        parsed_args.get_string(0),
        parsed_args.get_int64_or_else(1, Graph::DEFAULT_GRAPH_INITIAL_CAPACITY))
        .id());
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyPushGraphCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "push_graph_ctx(int graph_id)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    Graph::push_graph_ctx(parsed_args.get_int64(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyPopGraphCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "pop_op_ctx()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    Graph::pop_graph_ctx();
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyGraphCtx_methods[] = {
  {"get_graph", (PyCFunction) PyGraph_get_graph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"delete_graph", (PyCFunction) PyGraph_delete_graph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"get_default_define_and_run_graph", (PyCFunction) PyGraph_get_default_define_and_run_graph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"get_default_define_by_run_graph", (PyCFunction) PyGraph_get_default_define_by_run_graph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"get_default_eager_graph", (PyCFunction) PyGraph_get_default_eager_graph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"make_new_eager_graph", (PyCFunction) PyGraph_make_new_eager_graph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"make_new_define_and_run_graph", (PyCFunction) PyGraph_make_new_define_and_run_graph, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"push_graph_ctx", (PyCFunction) PyPushGraphCtx, METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pop_graph_ctx", (PyCFunction) PyPopGraphCtx, METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE
PyGetSetDef PyGraph_properties[] = {
  {PY_GET_SET_DEF_NAME("name"), (getter) PyGraph_name, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("id"), (getter) PyGraph_id, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("num_strategy"), (getter) PyGraph_num_strategy, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("cur_strategy_id"), (getter) PyGraph_cur_strategy_id, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("use_hetero_id"), (getter) PyGraph_use_hetero_id, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("cur_hetero_id"), (getter) PyGraph_cur_hetero_id, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyGraph_methods[] = {
  {"run", (PyCFunction) PyGraph_run, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"set_num_strategy", (PyCFunction) PyGraph_set_num_strategy, METH_VARARGS | METH_KEYWORDS, nullptr },
  {"merge_strategy", (PyCFunction) PyGraph_merge_strategy, METH_VARARGS | METH_KEYWORDS, nullptr },  
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyGraph_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.Graph", /* tp_name */
  sizeof(PyGraph), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyGraph_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyGraph_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyGraph_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  nullptr, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  PyGraph_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyGraph_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  nullptr, /* tp_new */
};
PyTypeObject* PyGraph_Type = &PyGraph_Type_obj;

void AddPyGraphTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyGraph_Type) < 0) 
    << "PyGraph_Type not ready";
  Py_INCREF(PyGraph_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "Graph", reinterpret_cast<PyObject*>(PyGraph_Type)))
    << "Failed to add PyGraph_Type";
}

void AddGraphContextManagingFunctionsToModule(py::module_& m) {
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(m.ptr(), PyGraphCtx_methods))
    << "Failed to add graph context managing methods";
}

} // namespace graph
} // namespace hetu
