/*********************************************************************************
*     File Name           :     sparse_vector.h
*     Created By          :     yuewu
*     Description         :
**********************************************************************************/

#ifndef LSOL_MATH_SPARSE_VECTOR_H__
#define LSOL_MATH_SPARSE_VECTOR_H__

#include <utility>

#include <lsol/math/shape.h>
#include <lsol/math/matrix_storage.h>
#include <lsol/math/matrix_expression.h>

namespace lsol {
namespace math {

template <typename DType>
class SVector
    : public expr::MatrixExp<SVector<DType>, DType, expr::ExprType::kSparse> {
 public:
  /// \brief  constructor
  SVector()
      : indexes_(nullptr), values_(nullptr), shape_(nullptr), count_(nullptr) {}

  /// \brief  copy constructor from another array, note the constructed
  /// array will  point to the same memory space
  ///
  /// \param src_vec Source matrix
  SVector(const SVector<DType>& src_vec)
      : indexes_(src_vec.indexes_),
        values_(src_vec.values_),
        shape_(src_vec.shape_),
        count_(src_vec.count_) {
    if (this->count_ != nullptr) ++(*this->count_);
  }

  /// \brief  destructor
  virtual ~SVector() { this->release(); }

  /// \brief assignment from value
  inline SVector<DType>& operator=(const DType& val) {
    return this->assign(val);
  }

  /// \brief assignment from matrix
  inline SVector<DType>& operator=(const SVector<DType>& src_vec) {
    if (this->count_ != src_vec.count_) {
      this->release();
      this->indexes_ = src_vec.indexes_;
      this->values_ = src_vec.values_;
      this->shape_ = src_vec.shape_;
      this->count_ = src_vec.count_;
      if (this->count_ != nullptr) ++(*this->count_);
    }
    return *this;
  }

 private:
  /// \brief init the owned data
  inline void init() {
    if (this->count_ == nullptr) {
      this->indexes_ = new MatrixStorage<index_t>;
      this->values_ = new MatrixStorage<DType>;
      this->shape_ = new Shape<2>;
      this->count_ = new int;
      *(this->count_) = 1;
    }
  }
  /// \brief  Release the spaces and counters
  void release() {
    if (this->count_ != nullptr) {
      --(*this->count_);
      if (*this->count_ == 0) {
        DeletePointer(this->indexes_);
        DeletePointer(this->values_);
        DeletePointer(this->shape_);
        DeletePointer(this->count_);
      }
    }
  }

 public:
  /// \brief  Reserver the specified space for matrix, the size of the matrix
  /// will be kept the same, but the capacity of the new matrix will exactly be
  /// the specified size
  ///
  /// \param new_size New number of elements to reserve
  inline void reserve(size_t new_size) {
    this->indexes_->resize(new_size);
    this->values_->resize(new_size);
  }

  /// \brief  reshape the array to the given shape, not the capacity is not
  /// ensured to be equal to the new size
  ///
  /// \param new_size New size to resize to
  void resize(size_t new_size) {
    this->init();
    static size_t max_size = 1 << 30;

    if (this->capacity() < new_size) {
      // allocate more memory
      size_t alloc_size = this->capacity();
      do {
        alloc_size += (alloc_size < max_size ? alloc_size : max_size) + 3;
      } while (alloc_size < new_size);

      this->indexes_->resize(alloc_size);
      this->values_->resize(alloc_size);
    }
    (*this->shape_)[1] = new_size;
  }

  /// \brief  Push a new element to the end of the vector, resize the array
  /// accordingly
  ///
  /// \param idx index
  /// \param val value
  inline void push_back(size_t idx, DType val) {
    size_t sz = this->size();
    this->resize(sz + 1);
    this->index(sz) = idx;
    this->value(sz) = val;
  }

  inline void clear() { this->resize(0); }

 public:
  inline size_t capacity() const { return this->indexes_->size(); }

  /// \brief  number of elements
  inline size_t size() const {
    return this->shape_ == nullptr ? 0 : (*this->shape_)[1];
  }
  inline index_t dim() const {
    size_t sz = this->size();
    return sz == 0 ? 0 : this->index(sz - 1) + 1;
  }

  inline const Shape<2>& shape() const { return *this->shape_; }

  inline bool empty() const {
    return this->shape_ == nullptr || this->size() == 0;
  }

  /// accessing elements
  inline index_t* indexes() { return this->indexes_->begin(); }
  inline const index_t* indexes() const { return this->indexes_->begin(); }
  inline index_t& index(size_t idx) { return this->indexes_->begin()[idx]; }
  inline const index_t& index(size_t idx) const {
    return this->indexes_->begin()[idx];
  }

  inline DType* values() { return this->values_->begin(); }
  inline const DType* values() const { return this->values_->begin(); }
  inline DType& value(size_t idx) { return this->values_->begin()[idx]; }
  inline const DType& value(size_t idx) const {
    return this->values_->begin()[idx];
  }

  inline std::pair<index_t, DType> operator[](size_t idx) const {
    return std::make_pair<index_t, DType>(this->index(idx), this->value(idx));
  }

  template <typename DType2>
  friend std::ostream& operator<<(std::ostream& os,
                                  const SVector<DType2>& svec);

 protected:
  MatrixStorage<index_t>* indexes_;
  MatrixStorage<DType>* values_;
  Shape<2>* shape_;
  int* count_;
};

template <typename DType>
std::ostream& operator<<(std::ostream& os, const SVector<DType>& svec) {
  os << "[ ";
  size_t sz = svec.size();
  for (size_t i = 0; i < sz; ++i) {
    os << svec.index(i) << ":" << svec.value(i) << " ";
  }
  os << "]\n";
  return os;
}

}  // namespace math
}  // namespace lsol

#endif
