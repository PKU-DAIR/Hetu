#include "hetu/core/symbol.h"
#include "hetu/common/except.h"

namespace hetu {

template class SymbolDef<int64_t>;
template class Symbol<int64_t>;

template <typename T>
T SymbolDef<T>::get_val() const {
  if (_input_1 != nullptr && _input_2 != nullptr) {
    switch (_op) {
      case SymbolOp::ADD: 
        return _input_1->get_val() + _input_2->get_val();
      case SymbolOp::SUB: 
        return _input_1->get_val() - _input_2->get_val();
      case SymbolOp::MUL: 
        return _input_1->get_val() * _input_2->get_val();
      case SymbolOp::DIV: {
        auto divisor = _input_2->get_val();
        HT_ASSERT(divisor) << "DIV op can't divide 0";
        return _input_1->get_val() / divisor;
      }
      case SymbolOp::REM: {
        HT_ASSERT((std::is_same<T, int64_t>::value)) << "REM op could only used for int64_t";
        auto divisor = _input_2->get_val();
        HT_ASSERT(divisor) << "REM op can't divide 0";
        return _input_1->get_val() % divisor;
      }
      default:
        HT_RUNTIME_ERROR << "SymbolOp type unspported!";
    }
  }
  HT_ASSERT(_input_1 == nullptr && _input_2 == nullptr) << "Something wrong when initializing the Symbol";
  HT_ASSERT(_is_leaf) << "Something wrong when initializing the Symbol";
  HT_ASSERT(is_instantiated()) << "Please ensure all the related Symbol is instantiated before get_val()";
  return _val;
}

bool is_SyShape_leaf(const SyShape& sy_shape) {
  for (const auto& x : sy_shape) {
    if (!x->is_leaf())
      return false;
  }
  return true;
}

HTShape get_HTShape_from_SyShape(const SyShape& sy_shape) {
  HTShape shape;
  for (const auto& x : sy_shape) {
    shape.push_back(std::move(x->get_val()));
  }
  return shape;
}

void set_HTShape_to_SyShape(const HTShape& ht_shape, SyShape& sy_shape) {
  HT_ASSERT(ht_shape.size() == sy_shape.size())
    << "the HTShape and SyShape should have equal dims, but got "
    << ht_shape.size() << " v.s. " << sy_shape.size();
  auto len = ht_shape.size();
  for (size_t i = 0; i < len; i++) {
    sy_shape[i] = ht_shape[i];
  }
}

std::ostream& operator << (std::ostream& os, const SyShape& sy_shape) {
  os << "SyShape";
  os << "(size=" << sy_shape.size();
  for (const auto& x : sy_shape) {
    if (x.is_defined()) {
      if (x->is_leaf())
        os << ", LEAF";
      else
        os << ", " << x->get_op_name();
    }
  }
  os << ")";
  return os;
}

template <typename T>
std::string Symbol<T>::symbol_info() const {
      std::ostringstream os;
      os << "Symbol(";
      if (this->_ptr != nullptr) {
        if (this->_ptr->is_leaf())
          os << "LEAF";
        else
          os << this->_ptr->get_op_name();
      } else {
        os << "NULL";
      }
      os << ")";
      return os.str(); 
    }

} // namespace hetu
