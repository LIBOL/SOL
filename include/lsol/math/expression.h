/*********************************************************************************
*     File Name           :     expression.h
*     Created By          :     yuewu
*     Description         :     expression base templates
**********************************************************************************/
#ifndef LSOL_MATH_EXPRESSION_H__
#define LSOL_MATH_EXPRESSION_H__

#include <stdexcept>

#include <lsol/math/operator.h>
#include <lsol/math/shape.h>

namespace lsol {
namespace math {
namespace expr {

/// \brief  type of expression
namespace ExprType {
/// \brief  value expression
const int kValue = 0;
/// \brief  dense expression
const int kDense = 1;
/// \brief  sparse expression
const int kSparse = 2;
}

/// \brief  base class for expression templates
///
/// \tparam SubType inherited class type
/// \tparam DType data type of each element in the expression
/// \tparam expr_type expression type
template <typename SubType, typename DType, int expr_type>
struct Exp {
  /// \brief  pointer to the current class instance
  inline SubType *selfptr() { return static_cast<SubType *>(this); }
  inline const SubType *selfptr() const {
    auto tmp = static_cast<const SubType *>(this);
    return static_cast<const SubType *>(this);
  }
  /// \brief  subtype instance of the current class
  inline SubType &self() { return *(static_cast<SubType *>(this)); }
  inline const SubType &self() const {
    return *(static_cast<const SubType *>(this));
  }

  inline int type() const { return expr_type; }
};

/// \brief  expression for value classes
///
/// \tparam CType class type
/// \tparam DType element data type
template <typename CType, typename DType>
struct MatrixBaseExp;

template <typename OP, typename CType, typename DType, typename EType,
          int expr_type>
void CalcExp(MatrixBaseExp<CType, DType> &dst,
             const Exp<EType, DType, expr_type> exp);

/// \brief  engine to execute the expressions, this engine works to map from
// expression templates to class-specific operations
///
/// \tparam OP operator
/// \tparam CType the actual class type to save the result
/// \tparam DType element data type of CType
template <typename OP, typename CType, typename DType>
struct ExpEngine {
  template <typename EType, int expr_type>
  inline static void Calc(CType &dst, const Exp<EType, DType, expr_type> &exp) {
    CalcExp<OP>(dst, exp);
  }
};

//---------------
// common expressions
// --------------

/// \brief  scalar expression
///
/// \tparam DType data type of the scalar
template <typename DType>
struct ScalarExp : public Exp<ScalarExp<DType>, DType, ExprType::kValue> {
  DType value;
  /// \brief  implicit constructor, MUST NOT BE explicit
  ScalarExp(DType scalar) : value(scalar) {}

  inline DType operator()(size_t x, size_t y) const { return value; }
  inline DType operator[](size_t idx) const { return value; }

  inline Shape<2> shape() const { return Shape<2>(); }
};

/// \brief  make a scalar expression
template <typename DType>
inline ScalarExp<DType> MakeExp(DType val) {
  return ScalarExp<DType>(val);
}

//---------------
// BinaryMapExp
// --------------
/// \brief  binary map expression lhs [op] rhs
///
/// \tparam OP operator
/// \tparam EType1 expression type of lhs
/// \tparam EType2 expression type of rhs
/// \tparam DType data element type
template <typename OP, typename EType1, typename EType2, typename DType,
          int exptype>
struct BinaryMapExp
    : public Exp<BinaryMapExp<OP, EType1, EType2, DType, exptype>, DType,
                 exptype> {
  /// \brief  left operand
  const EType1 lhs;
  /// \brief  right operand
  const EType2 rhs;
  /*! \brief constructor */
  BinaryMapExp(const EType1 &_lhs, const EType2 &_rhs) : lhs(_lhs), rhs(_rhs) {}

  /// accessing elements
  inline DType operator()(size_t x, size_t y) const {
    return OP::map(lhs(x, y), rhs(x, y));
  }
  inline DType operator[](size_t idx) const {
    return OP::map(lhs[idx], rhs[idx]);
  }

  inline Shape<2> shape() const {
    const Shape<2> &shape1 = lhs.shape();
    const Shape<2> &shape2 = rhs.shape();
    if (shape1.size() == 0) return shape2;
    if (shape2.size() == 0) return shape1;
    if (shape1 == shape2) {
      return shape1;
    }

    throw std::runtime_error("BinaryMapExp: shapes of operands are different");
  }
};

/// \brief  make a binary expression
template <typename OP, typename EType1, typename EType2, typename DType,
          int exptype1, int exptype2>
inline BinaryMapExp<OP, EType1, EType2, DType, (exptype1 | exptype2)> MakeExp(
    const Exp<EType1, DType, exptype1> &lhs,
    const Exp<EType2, DType, exptype2> &rhs) {
  return BinaryMapExp<OP, EType1, EType2, DType, (exptype1 | exptype2)>(
      lhs.self(), rhs.self());
}

/// arithmetic operator expressions
#define ArithmeticBinaryMapExpTpl(opx, opr)                                  \
  template <typename EType1, typename EType2, typename DType, int exptype1,  \
            int exptype2>                                                    \
  inline auto operator opx(const Exp<EType1, DType, exptype1> &lhs,          \
                           const Exp<EType2, DType, exptype2> &rhs)          \
      ->decltype(MakeExp<op::opr>(lhs, rhs)) {                               \
    return MakeExp<op::opr>(lhs, rhs);                                       \
  }                                                                          \
                                                                             \
  template <typename EType, typename DType, int exptype>                     \
  inline auto operator opx(const Exp<EType, DType, exptype> &lhs, DType rhs) \
      ->decltype(lhs opx MakeExp<DType>(rhs)) {                              \
    return lhs opx MakeExp<DType>(rhs);                                      \
  }                                                                          \
                                                                             \
  template <typename EType, typename DType, int exptype>                     \
  inline auto operator opx(DType lhs, const Exp<EType, DType, exptype> &rhs) \
      ->decltype(MakeExp<DType>(lhs) opx rhs) {                              \
    return MakeExp<DType>(lhs) opx rhs;                                      \
  }

ArithmeticBinaryMapExpTpl(+, plus);
ArithmeticBinaryMapExpTpl(-, minus);
ArithmeticBinaryMapExpTpl(*, mul);
ArithmeticBinaryMapExpTpl(/, div);

//---------------
// UnaryMapExp
// --------------

/// \brief  unary map expression op(src)
///
/// \tparam OP operator
/// \tparam EType expression type
/// \tparam DType data element type
template <typename OP, typename EType, typename DType, int exptype>
struct UnaryMapExp
    : public Exp<UnaryMapExp<OP, EType, DType, exptype>, DType, exptype> {
  /*! \brief source expression */
  const EType &exp;
  /*! \brief constructor */
  explicit UnaryMapExp(const EType &src) : exp(src) {}
};

/*! \brief make expression */
template <typename OP, typename EType, typename DType, int exptype>
inline UnaryMapExp<OP, EType, DType, exptype> MakeExp(
    const Exp<EType, DType, exptype> &src) {
  return UnaryMapExp<OP, EType, DType>(src.self());
}

}  // namespace expr
}  // namespace math
}  // namespace lsol
#endif
