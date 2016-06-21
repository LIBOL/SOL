/*********************************************************************************
 *     File Name           :     vector.h
 *     Created By          :     yuewu
 *     Description         :
 **********************************************************************************/
#ifndef LSOL_MATH_VECTOR_H__
#define LSOL_MATH_VECTOR_H__

#include <lsol/math/matrix.h>
#include <functional>

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
  inline void resize(size_t new_size) {
    this->init();
    static size_t max_size = 1 << 30;
    if (this->capacity() < new_size) {
      // allocate more memory
      size_t alloc_size = this->capacity();
      do {
        alloc_size += (alloc_size < max_size ? alloc_size : max_size) + 3;
      } while (alloc_size < new_size);

      this->storage_->resize(alloc_size);
    }
    (*this->shape_)[0] = 1;
    (*this->shape_)[1] = new_size;
  }

  /// \brief  Push a new element to the end of the vector, resize the array
  /// accordingly
  ///
  /// \param elem Element to be pushed
  inline void push_back(const DType& elem) {
    this->resize(this->size() + 1);
    this->back() = elem;
  }

  /// \brief  Pop out the last element, work like a stack
  inline void pop_back() {
    size_t sz = this->size();
    if (sz == 0) {
      throw std::runtime_error("pop back failed as vector is empty!");
    }
    this->resize(sz - 1);
  }

  /// \brief  Resize the array to be of size zero
  inline void clear(void) { this->resize(0); }

  inline void slice_op(const std::function<void(DType&)>& op, size_t start = 0,
                       size_t end = -1) {
    DType* start_iter = this->begin() + start;
    DType* end_iter = end == -1 ? this->end() : this->begin() + end;
    for (DType* iter = start_iter; iter != end_iter; ++iter) op(*iter);
  }

 public:
  inline size_t dim() const {
    return this->shape_ == nullptr ? 0 : (*this->shape_)[1];
  }
  inline size_t size() const { return this->dim(); }

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
