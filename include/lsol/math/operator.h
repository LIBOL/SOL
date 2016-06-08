/*********************************************************************************
*     File Name           :     operator.h
*     Created By          :     yuewu
*     Description         :     operators on expressions
**********************************************************************************/

#ifndef LSOL_MATH_OPERATOR_H__
#define LSOL_MATH_OPERATOR_H__

namespace lsol {
namespace math {
namespace expr {
namespace op {

enum class OpType { kNone = 0, kL1 = 1, kL2 = 2 };

//---------------
// binary operators
// --------------
struct plus {
  template <typename DType>
  inline static DType map(const DType& a, const DType& b) {
    return a + b;
  }
};

struct minus {
  template <typename DType>
  inline static DType map(const DType& a, const DType& b) {
    return a - b;
  }
};

struct mul {
  template <typename DType>
  inline static DType map(const DType& a, const DType& b) {
    return a * b;
  }
};

struct div {
  template <typename DType>
  inline static DType map(const DType& a, const DType& b) {
    return a / b;
  }
};

struct max {
  template <typename DType>
  inline static DType map(const DType& a, const DType& b) {
    return a > b ? a : b;
  }
};

struct min {
  template <typename DType>
  inline static DType map(const DType& a, const DType& b) {
    return a < b ? a : b;
  }
};

struct assign {
  template <typename DType>
  inline static void map(DType& a, const DType& b) {
    a = b;
  }
};

struct plusto {
  template <typename DType>
  inline static void map(DType& a, const DType& b) {
    a += b;
  }
};

struct minusto {
  template <typename DType>
  inline static void map(DType& a, const DType& b) {
    a -= b;
  }
};

struct multo {
  template <typename DType>
  inline static void map(DType& a, const DType& b) {
    a *= b;
  }
};

struct divto {
  template <typename DType>
  inline static void map(DType& a, const DType& b) {
    a /= b;
  }
};

//---------------
// unary operators
// --------------
struct abs {
  template <typename DType>
  inline static DType map(const DType& a) {
    return a > 0 ? a : -a;
  }
};

struct square {
  template <typename DType>
  inline static DType map(const DType& a) {
    return a * a;
  }
};

}  // namespace op
}  // namespace expr
}  // namespace math
}  // namespace lsol
#endif
