/*********************************************************************************
 *     File Name           :     vector.h
 *     Created By          :     yuewu
 *     Description         :
 **********************************************************************************/
#ifndef LSOL_MATH_VECTOR_H__
#define LSOL_MATH_VECTOR_H__

#include <lsol/math/matrix.h>

namespace lsol {
namespace math {

template <typename DType>
class Vector : public Matrix<DType> {
 public:
  /// \brief  constructors
  Vector() : Matrix<DType>() {}
  Vector(size_t size) : Matrix<DType>({1, size}) {}

  /// \brief  copy constructor from an expression template
  ///
  /// \tparam EType <++>
  /// \tparam etype <++>
  /// \param exp <++>
  template <typename EType, int etype>
  Vector(const expr::Exp<EType, DType, etype>& exp)
      : Matrix<DType>(exp) {}

  /// \brief  copy constructors
  Vector(const Vector<DType>& src_vec) : Matrix<DType>(src_vec) {}

  /// \brief  destructor
  virtual ~Vector() {}

  /// \brief assignment from an expression template
  template <typename EType, int etype>
  Vector<DType>& operator=(const expr::Exp<EType, DType, etype>& exp) {
    return static_cast<Vector<DType>&>(Matrix<DType>::operator=(exp));
  }

  /// \brief assignment from value
  inline Vector<DType>& operator=(const DType& val) {
    return static_cast<Vector<DType>&>(Matrix<DType>::operator=(val));
  }

  /// \brief assignment from matrix
  inline Vector<DType>& operator=(const Vector<DType>& src_vec) {
    return static_cast<Vector<DType>&>(Matrix<DType>::operator=(src_vec));
  }

 public:
  /// \brief  Push a new element to the end of the vector, resize the array
  /// accordingly
  ///
  /// \param elem Element to be pushed
  inline void push_back(const DType& elem) {
    this->resize({1, this->size() + 1});
    this->back() = elem;
  }

  /// \brief  Pop out the last element, work like a stack
  inline void pop_back() {
    size_t size = this->size();
    if (size == 0) {
      throw std::runtime_error("pop back failed as vector is empty!");
    }
    this->resize({1, size - 1});
  }

  /// \brief  Resize the array to be of size zero
  inline void clear(void) { this->resize({0, 0}); }

 public:
  inline size_t dim() const {
    return this->shape_ == nullptr ? 0 : (*this->shape_)[1];
  }

  inline DType* begin() { return this->storage_->begin(); }
  inline const DType* begin() const { return this->storage_->begin(); }

  inline DType* end() { return this->storage_->begin() + this->size(); }
  inline const DType* end() const {
    return this->storage_->begin() + this->size();
  }

  inline const DType& front() const { return *(this->begin()); }
  inline DType& front() { return *(this->begin()); }

  inline const DType& back() const { return *(this->end() - 1); }
  inline DType& back() { return *(this->end() - 1); }
};

}  // namespace math
}  // namespace lsol
#endif
