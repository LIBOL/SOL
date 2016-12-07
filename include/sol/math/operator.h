/*********************************************************************************
*     File Name           :     operator.h
*     Created By          :     yuewu
*     Description         :     operators on expressions
**********************************************************************************/

#ifndef SOL_MATH_OPERATOR_H__
#define SOL_MATH_OPERATOR_H__

#include <cmath>

namespace sol {
namespace math {
namespace expr {
namespace op {

enum OpType { kNone = 0, kL1 = 1, kL2 = 2 };

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

// truncate
struct trunc {
  template <typename DType>
  inline static DType map(const DType& a, const DType& b) {
    if (a >= 0)
      return a > b ? a - b : 0;
    else
      return -a > b ? a + b : 0;
  }
};

// get the left value
struct left {
  template <typename DType>
  inline static const DType& map(const DType& a, const DType& b) {
    return a;
  }
};

// get the right value
struct right {
  template <typename DType>
  inline static const DType& map(const DType& a, const DType& b) {
    return b;
  }
};

struct smaller {
  template <typename DType>
  inline static bool map(const DType& a, const DType& b) {
    return a < b;
  }
};

struct larger {
  template <typename DType>
  inline static bool map(const DType& a, const DType& b) {
    return a > b;
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

struct sqrt {
  template <typename DType>
  inline static DType map(const DType& a) {
    return sqrtf(a);
  }
};

}  // namespace op
}  // namespace expr
}  // namespace math
}  // namespace sol
#endif
