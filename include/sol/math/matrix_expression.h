/*********************************************************************************
*     File Name           :     matrix_expression.h
*     Created By          :     yuewu
*     Description         :     expression templates for matrix
**********************************************************************************/

#ifndef SOL_MATH_MATRIX_EXPRESSION_H__
#define SOL_MATH_MATRIX_EXPRESSION_H__

#include <sol/math/expression.h>

namespace sol {
namespace math {

namespace expr {

//---------------
// MatrixExp
// --------------

/// \brief  base class for dense matrices this expression defines the
/// basic arithmetic operators on dense matrices classes
///
/// \tparam CType the actual class type
/// \tparam DType element data type
template <typename CType, typename DType, int exptype>
struct MatrixExp : public Exp<CType, DType, exptype> {
  /*! Arithmetic operations with scalars*/
  inline CType &operator+=(const DType &val) {
    ExpEngine<op::plusto, CType, DType>::Calc(this->self(),
                                              MakeExp<DType>(val));
    return this->self();
  }
  inline CType &operator-=(const DType &val) {
    ExpEngine<op::minusto, CType, DType>::Calc(this->self(),
                                               MakeExp<DType>(val));
    return this->self();
  }
  inline CType &operator*=(const DType &val) {
    ExpEngine<op::multo, CType, DType>::Calc(this->self(), MakeExp<DType>(val));
    return this->self();
  }
  inline CType &operator/=(const DType &val) {
    ExpEngine<op::divto, CType, DType>::Calc(this->self(), MakeExp<DType>(val));
    return this->self();
  }

  inline CType &assign(const DType &val) {
    ExpEngine<op::assign, CType, DType>::Calc(this->self(),
                                              MakeExp<DType>(val));
    return this->self();
  }

  /*! Arithmetic operations with expressions*/
  template <typename CType2, typename DType2, int etype, int continuous>
  inline CType &operator+=(const Exp<CType2, DType2, etype, continuous> &exp) {
    ExpEngine<op::plusto, CType, DType>::Calc(this->self(), exp.self());
    return this->self();
  }
  template <typename CType2, typename DType2, int etype, int continuous>
  inline CType &operator-=(const Exp<CType2, DType2, etype, continuous> &exp) {
    ExpEngine<op::minusto, CType, DType>::Calc(this->self(), exp.self());
    return this->self();
  }
  template <typename CType2, typename DType2, int etype, int continuous>
  inline CType &operator*=(const Exp<CType2, DType2, etype, continuous> &exp) {
    ExpEngine<op::multo, CType, DType>::Calc(this->self(), exp.self());
    return this->self();
  }
  template <typename CType2, typename DType2, int etype, int continuous>
  inline CType &operator/=(const Exp<CType2, DType2, etype, continuous> &exp) {
    ExpEngine<op::divto, CType, DType>::Calc(this->self(), exp.self());
    return this->self();
  }

  template <typename CType2, typename DType2, int etype, int continuous>
  inline CType &assign(const Exp<CType2, DType2, etype, continuous> &exp) {
    ExpEngine<op::assign, CType, DType>::Calc(this->self(), exp.self());
    return this->self();
  }
};

/*
template <typename OP, typename MatType, int kDim, typename DType,
          typename EType, typename DType2, ExprType expr_type2>
void CalcExp(MatrixBase<MatType, DType, kDim> &dst,
             const Exp<EType, DType2, expr_type2> &exp) {
  throw std::runtime_error("invalid operation occured");
}
*/

// value operations
template <typename OP, typename EType, typename DType, int exptype>
inline DType reduce(const Exp<EType, DType, ExprType::kValue> &exp) {
  return exp.self().value_;
}

// dense matrix operations
template <typename OP, typename CType, typename DType, typename EType,
          int exptype>
void CalcExp(MatrixExp<CType, DType, ExprType::kDense> &dst,
             const Exp<EType, DType, exptype> &exp) {
  CType &mat = dst.self();
  const EType &exp_val = exp.self();
  // shape check
  const Shape<2> s = mat.shape();
  const Shape<2> s1 = exp_val.shape();
  if (s1.size() != 0 && s != s1) {
    throw std::runtime_error("matrix operation on different shape");
  }
  size_t sz = s.size();
  DType *pdata = mat.data();
  for (size_t idx = 0; idx < sz; ++idx) {
    OP::template map<DType>(*pdata++, exp_val[idx]);
  }
}

template <typename OP, typename CType, typename DType, typename EType,
          int exptype>
void CalcExp(MatrixExp<CType, DType, ExprType::kDense> &dst,
             const Exp<EType, DType, exptype, 0> &exp) {
  CType &mat = dst.self();
  const EType &exp_val = exp.self();
  // shape check
  const Shape<2> s = mat.shape();
  const Shape<2> s1 = exp_val.shape();
  if (s1.size() != 0 && s != s1) {
    throw std::runtime_error("matrix operation on different shape");
  }
  size_t rows = s[0];
  size_t cols = s[1];
  DType *pdata = mat.data();
  for (size_t y = 0; y < rows; ++y) {
    for (size_t x = 0; x < cols; ++x) {
      OP::template map<DType>(*pdata++, exp_val(y, x));
    }
  }
}

template <typename OP, typename EType1, typename EType2, typename DType>
inline DType dot(const Exp<EType1, DType, ExprType::kDense> &lhs,
                 const Exp<EType2, DType, ExprType::kDense> &rhs) {
  const EType1 &exp_val1 = lhs.self();
  const EType2 &exp_val2 = rhs.self();
  // shape check
  const Shape<2> s1 = exp_val1.shape();
  const Shape<2> s2 = exp_val2.shape();
  if (s1 != s2) {
    throw std::runtime_error("matrix dot on different shape");
  }
  size_t sz = s1.size();
  DType val = 0;
  for (size_t idx = 0; idx < sz; ++idx) {
    val += OP::template map<DType>(exp_val1[idx], exp_val2[idx]);
  }
  return val;
}

template <typename OP, typename EType, typename DType>
inline DType reduce(const Exp<EType, DType, ExprType::kDense> &exp) {
  const EType &exp_val = exp.self();
  const Shape<2> s = exp_val.shape();
  size_t sz = s.size();
  DType val = 0;
  for (size_t idx = 0; idx < sz; ++idx) {
    val = OP::template map<DType>(val, exp_val[idx]);
  }
  return val;
}

// dense matrix with sparse matrix operations
template <typename OP, typename CType, typename DType, typename EType>
void CalcExp(MatrixExp<CType, DType, ExprType::kDense> &dst,
             const Exp<EType, DType, ExprType::kSparse> &exp) {
  const CType &dvec = dst.self();
  size_t dsz = dvec.shape().size();
  DType *pdata = dst.self().data();

  const EType &svec = exp.self();
  size_t sz = svec.shape().size();
  while (sz > 0 && svec.index(sz - 1) >= dsz) --sz;

  for (size_t idx = 0; idx < sz; ++idx) {
    OP::template map<DType>(pdata[svec.index(idx)], svec.value(idx));
  }
}

template <typename OP, typename EType1, typename EType2, typename DType>
inline DType dot(const Exp<EType1, DType, ExprType::kDense> &lhs,
                 const Exp<EType2, DType, ExprType::kSparse> &rhs) {
  const EType1 &exp_val1 = lhs.self();
  const EType2 &exp_val2 = rhs.self();
  size_t lhs_sz = exp_val1.shape().size();
  size_t sz = exp_val2.shape().size();
  while (sz > 0 && exp_val2.index(sz - 1) >= lhs_sz) --sz;
  DType val = 0;
  for (size_t idx = 0; idx < sz; ++idx) {
    val += OP::template map<DType>(exp_val1[exp_val2.index(idx)],
                                   exp_val2.value(idx));
  }
  return val;
}

// sparse matrix operations
template <typename OP, typename CType, typename DType, typename EType>
void CalcExp(MatrixExp<CType, DType, ExprType::kSparse> &dst,
             const Exp<EType, DType, ExprType::kValue> &exp) {
  CType &mat = dst.self();
  const EType &exp_val = exp.self();
  size_t sz = mat.size();
  DType *pdata = mat.values().data();
  for (size_t idx = 0; idx < sz; ++idx) {
    OP::template map<DType>(*pdata++, exp_val[idx]);
  }
}

template <typename OP, typename EType, typename DType>
inline DType reduce(const Exp<EType, DType, ExprType::kSparse> &exp) {
  const EType &exp_val = exp.self();
  // shape check
  const Shape<2> s = exp_val.shape();
  size_t sz = s.size();
  DType val = 0;
  for (size_t idx = 0; idx < sz; ++idx) {
    val = OP::template map<DType>(val, exp_val.value(idx));
  }
  return val;
}

template <typename EType, typename DType, int exptype>
DType Norm1(const Exp<EType, DType, exptype> &src) {
  return reduce<op::plus>(L1(src.self()));
}

template <typename EType, typename DType, int exptype>
DType Norm2(const Exp<EType, DType, exptype> &src) {
  return reduce<op::plus>(L2(src.self()));
}

}  // namespace expr
}  // namespace math
}  // namespace sol

#endif
