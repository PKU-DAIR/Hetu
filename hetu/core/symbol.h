#pragma once

#include "hetu/utils/shared_ptr_wrapper.h"
#include "hetu/core/ndarray_meta.h"
#include <vector>
#include <string>

namespace hetu {

enum class SymbolOp {
  ADD = 0,
  SUB,
  MUL,
  DIV,
  REM
};

template <typename T>
class SymbolDef : public shared_ptr_target  {
  private:
    bool _is_leaf = false; // SymbolDef(T _val) is the only way to make it a leaf
    bool _is_instantiated = false; // SymbolDef(T _val) and set_val() are the only two ways to instantiate 
    T _val{};
    SymbolOp _op;
    std::shared_ptr<SymbolDef> _input_1;
    std::shared_ptr<SymbolDef> _input_2;

  public:
    SymbolDef(): _is_leaf(true) {
    };
    SymbolDef(T val): _is_leaf(true), _is_instantiated(true), _val(val) {
    }
    SymbolDef(const SymbolOp& op, const std::shared_ptr<SymbolDef>& input_1, const std::shared_ptr<SymbolDef>& input_2)
    : _op(op), _input_1(input_1), _input_2(input_2) {
    }
    SymbolDef(const SymbolOp& op, const SymbolDef& input_1, const SymbolDef& input_2)
    : _op(op), _input_1(std::make_shared<SymbolDef>(input_1)), _input_2(std::make_shared<SymbolDef>(input_2)) {
    }

    /*
    SymbolDef operator + (const SymbolDef& rhs) {    
      return SymbolDef(SymbolOp::ADD, *this, rhs);
    }
    SymbolDef operator - (const SymbolDef& rhs) {    
      return SymbolDef(SymbolOp::SUB, *this, rhs);
    }
    SymbolDef operator * (const SymbolDef& rhs) {    
      return SymbolDef(SymbolOp::MUL, *this, rhs);
    }
    SymbolDef operator / (const SymbolDef& rhs) {    
      return SymbolDef(SymbolOp::DIV, *this, rhs);
    }
    SymbolDef& operator = (const T& val) {  
      _val = val;  
      return *this;
    }
    */

    bool is_leaf() const {
      return _is_leaf;
    }

    bool is_instantiated() const {
      return _is_instantiated;
    }

    std::string get_op_name() const {
      switch (_op) {
        case SymbolOp::ADD:
          return "ADD";
        case SymbolOp::SUB:
          return "SUB";
        case SymbolOp::MUL:
          return "MUL";
        case SymbolOp::DIV:
          return "DIV";
        case SymbolOp::REM:
          HT_ASSERT((std::is_same<T, int64_t>::value)) << "REM op could only used for int64_t";
          return "REM";
        default:
          HT_RUNTIME_ERROR << "SymbolOp type unspported!";
      }
    }

    void set_val(T val) {
      HT_ASSERT(_is_leaf) << "Only leaf symbol can use set_val() method";
      _is_instantiated = true;
      _val = val;
    }

    T get_val() const;
};

template <typename T>
class Symbol : public shared_ptr_wrapper<SymbolDef<T>> {
  public:
    Symbol() = default;
    Symbol(const T& val)
    : shared_ptr_wrapper<SymbolDef<T>>() {
      this->_ptr = this->template make_ptr<SymbolDef<T>>(val);
    }
    Symbol(const SymbolOp& op, const std::shared_ptr<SymbolDef<T>>& input_1, const std::shared_ptr<SymbolDef<T>>& input_2)
    : shared_ptr_wrapper<SymbolDef<T>>() {
      this->_ptr = this->template make_ptr<SymbolDef<T>>(op, input_1, input_2);
    }
    Symbol(const SymbolOp& op, const SymbolDef<T>& input_1, const SymbolDef<T>& input_2)
    : shared_ptr_wrapper<SymbolDef<T>>() {
      this->_ptr = this->template make_ptr<SymbolDef<T>>(op, input_1, input_2);
    }

    const std::shared_ptr<SymbolDef<T>>& get_ptr() const {
      return this->_ptr;
    }

    void reset() {
      this->_ptr.reset();
    }

    explicit operator bool() const noexcept {
      return this->_ptr != nullptr;
    }

    bool operator == (const Symbol& other) const {
        return this->_ptr == other._ptr;
    }

    Symbol operator + (const Symbol& rhs) const {    
      return Symbol(SymbolOp::ADD, this->_ptr, rhs._ptr);
    }

    Symbol operator - (const Symbol& rhs) const {    
      return Symbol(SymbolOp::SUB, this->_ptr, rhs._ptr);
    }

    Symbol operator * (const Symbol& rhs) const {    
      return Symbol(SymbolOp::MUL, this->_ptr, rhs._ptr);
    }

    Symbol operator / (const Symbol& rhs) const {    
      return Symbol(SymbolOp::DIV, this->_ptr, rhs._ptr);
    }

    Symbol operator % (const Symbol& rhs) const {
      return Symbol(SymbolOp::REM, this->_ptr, rhs._ptr);
    }

    Symbol& operator = (const T& val) {
      if (this->_ptr != nullptr)
        this->_ptr->set_val(val); 
      else
        this->_ptr = this->template make_ptr<SymbolDef<T>>(val); 
      return *this;
    }

    std::string symbol_info() const;
};

using IntSymbol = Symbol<int64_t>;
using DoubleSymbol = Symbol<double>;
using SyShape = std::vector<IntSymbol>;
using SyShapeList = std::vector<SyShape>;

bool is_SyShape_leaf(const SyShape& sy_shape);
HTShape get_HTShape_from_SyShape(const SyShape& sy_shape);
void set_HTShape_to_SyShape(const HTShape& ht_shape, SyShape& sy_shape);

// Used only for debug
std::ostream& operator << (std::ostream& os, const SyShape& sy_shape);

} // namespace hetu

namespace std {
  template<>
  struct hash<hetu::IntSymbol> {
    size_t operator()(const hetu::IntSymbol& s) const noexcept {
      return hash<std::shared_ptr<hetu::SymbolDef<int64_t>>>()(s.get_ptr());
    }
  };
} // namespace std