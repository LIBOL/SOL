/*********************************************************************************
*     File Name           :     matrix_shape.h
*     Created By          :     yuewu
*     Description         :     shape definition of matrices
**********************************************************************************/
#ifndef LSOL_MATH_SHAPE_H__
#define LSOL_MATH_SHAPE_H__

#include <cstring>
#include <string>
#include <sstream>
#include <stdexcept>
#include <initializer_list>

#include <lsol/util/types.h>

namespace lsol {
namespace math {

template <int kDim>
class Shape {
 public:
  /// \brief  empty constructor
  Shape() { memset(this->shape_, 0, sizeof(size_t) * kDim); }

  Shape(const std::initializer_list<size_t>& shape) {
    if (shape.size() != kDim) {
      std::ostringstream oss;
      oss << "dimension of input shape (" << shape.size()
          << ") is compatible with the expected (" << kDim << ")";
      throw std::invalid_argument(oss.str());
    }
    size_t* sp = this->shape_;
    for (const size_t& s : shape) *sp++ = s;
  }

  /// \brief  copy constructor
  Shape(const Shape<kDim>& shape) {
    memcpy(this->shape_, shape.shape_, sizeof(size_t) * kDim);
  }

  Shape<kDim>& operator=(const Shape<kDim>& shape) {
    memcpy(this->shape_, shape.shape_, sizeof(size_t) * kDim);
    return *this;
  }

 public:
  inline Shape<2> FlatTo2D() const {
    Shape<2> s;
    s[1] = this->shape_[kDim - 1];
    s[0] = 1;
    for (int i = 0; i < kDim - 1; ++i) {
      s[0] *= this->shape_[i];
    }
    return s;
  }

 public:
  inline int dim() const { return kDim; }

  inline size_t& operator[](int idx) { return shape_[idx]; }
  inline const size_t& operator[](int idx) const { return shape_[idx]; }

  inline size_t size(int start = 0, int end = kDim) const {
    size_t sz = 1;
    for (int i = start; i < end; ++i) {
      sz *= this->shape_[i];
    }
    return sz;
  }

  inline size_t offset(int dim) const {
    size_t sz = 1;
    for (int i = dim + 1; i < kDim; ++i) {
      sz *= this->shape_[i];
    }
    return sz;
  }

  inline bool operator==(const Shape<kDim>& shape) const {
    for (int i = 0; i < kDim; ++i) {
      if (shape.shape_[i] != this->shape_[i]) return false;
    }
    return true;
  }
  inline bool operator!=(const Shape<kDim>& shape) const {
    return !(*this == shape);
  }

  template <int kDim2>
  friend std::ostream& operator<<(std::ostream& os, const Shape<kDim2>& s);

  inline std::string shape_string() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
  }

 protected:
  size_t shape_[kDim];
};

template <int kDim>
std::ostream& operator<<(std::ostream& os, const Shape<kDim>& s) {
  os << "shape: ";
  os << s.shape_[0];
  for (int i = 1; i < kDim; ++i) {
    os << "," << s.shape_[i];
  }
  return os;
}

}  // namespace math
}  // namespace lsol

#endif
