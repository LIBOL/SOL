/*********************************************************************************
*     File Name           :     matrix_storage.h
*     Created By          :     yuewu
*     Description         :
**********************************************************************************/

#ifndef LSOL_MATH_MATRIX_STORAGE_H__
#define LSOL_MATH_MATRIX_STORAGE_H__

#include <lsol/util/util.h>

namespace lsol {
namespace math {

/// \brief  storage structure of matrix
///
/// \tparam DType Data Element Type
template <typename DType>
class MatrixStorage {
 public:
  MatrixStorage() : begin_(nullptr), size_(0) {}

  ~MatrixStorage() { DeleteArray(this->begin_); }

 public:
  /// \brief  resize the storage
  ///
  /// \param new_size Specified size of elements to be allocated
  void resize(size_t new_size) {
    if (new_size > this->size_) {
      DType* new_begin = new DType[new_size];
      memset(new_begin, 0, sizeof(DType) * new_size);
      // copy data
      std::memcpy(new_begin, this->begin_, sizeof(DType) * this->size());
      DeleteArray(this->begin_);
      this->begin_ = new_begin;
      this->size_ = new_size;
    }
  }

  DISABLE_COPY_AND_ASSIGN(MatrixStorage);

 public:
  inline const DType* begin() const { return this->begin_; }
  inline DType* begin() { return this->begin_; }

  inline const DType* end() const { return this->begin_ + this->size_; }
  inline DType* end() { return this->end_ + this->size_; }

  inline size_t size() const { return this->size_; }

 protected:
  // point to the first element
  DType* begin_;
  // capacity of the array
  size_t size_;
};

}  // namespace math
}  // namespace lsol

#endif
