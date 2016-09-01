/*********************************************************************************
*     File Name           :     matrix.h
*     Created By          :     yuewu
*     Description         :     base structure of matrix
**********************************************************************************/
#ifndef SOL_MATH_MATRIX_H__
#define SOL_MATH_MATRIX_H__

#include <limits>

#include <sol/math/shape.h>
#include <sol/math/matrix_storage.h>
#include <sol/math/matrix_expression.h>

namespace sol {
namespace math {

template <typename DType>
class Matrix
    : public expr::MatrixExp<Matrix<DType>, DType, expr::ExprType::kDense> {
 public:
  /// \brief  constructor
  Matrix() : storage_(nullptr), shape_(nullptr), count_(nullptr) {}

  /// \brief  constructor
  Matrix(const Shape<2>& shape)
      : storage_(nullptr), shape_(nullptr), count_(nullptr) {
    this->init();
    this->resize(shape);
  }

  /// \brief  copy constructor from an expression template
  ///
  /// \tparam EType <++>
  /// \tparam etype <++>
  /// \param exp <++>
  template <typename EType, int etype>
  Matrix(const expr::Exp<EType, DType, etype>& exp)
      : storage_(nullptr), shape_(nullptr), count_(nullptr) {
    this->init();
    this->assign(exp);
  }

  /// \brief  copy constructor from another array, note the constructed
  /// array will  point to the same memory space
  ///
  /// \param src_mat Source matrix
  Matrix(const Matrix<DType>& src_mat)
      : storage_(src_mat.storage_),
        shape_(src_mat.shape_),
        count_(src_mat.count_) {
    if (this->count_ != nullptr) ++(*this->count_);
  }

  /// \brief  destructor
  virtual ~Matrix() { this->release(); }

  /// \brief assignment from an expression template
  template <typename EType, int etype>
  Matrix<DType>& operator=(const expr::Exp<EType, DType, etype>& exp) {
    this->init();
    this->assign(exp);
    return *this;
  }

  /// \brief assignment from value
  inline Matrix<DType>& operator=(const DType& val) {
    return this->assign(val);
  }

  /// \brief assignment from matrix
  inline Matrix<DType>& operator=(const Matrix<DType>& src_mat) {
    if (this->count_ != src_mat.count_) {
      this->release();
      this->storage_ = src_mat.storage_;
      this->shape_ = src_mat.shape_;
      this->count_ = src_mat.count_;
      if (this->count_ != nullptr) ++(*this->count_);
    }
    return *this;
  }

  inline void copyto(Matrix<DType>& dst_mat) {
    dst_mat.resize(this->shape());
    dst_mat.assign(*this);
  }

 protected:
  /// \brief init the owned data
  inline void init() {
    if (this->count_ == nullptr) {
      this->storage_ = new MatrixStorage<DType>;
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
        DeletePointer(this->storage_);
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
    this->init();
    this->storage_->resize(new_size);
  }

  /// \brief  reshape the array to the given shape, not the capacity is not
  /// ensured to be equal to the new size
  ///
  /// \param shape new shape of the matrix
  void resize(const Shape<2>& shape) {
    this->init();
    static size_t max_size = 1 << 30;

    size_t new_size = shape.size();
    if (this->capacity() < new_size) {
      // allocate more memory
      size_t alloc_size = this->capacity();
      do {
        alloc_size += (alloc_size < max_size ? alloc_size : max_size) + 3;
      } while (alloc_size < new_size);

      this->storage_->resize(alloc_size);
    }
    *(this->shape_) = shape;
  }

 public:
  inline size_t capacity() const { return this->storage_->size(); }

  /// \brief  number of elements
  inline size_t size(int start_dim = 0, int end_dim = 2) const {
    return this->shape_ == nullptr ? 0 : this->shape_->size(start_dim, end_dim);
  }
  inline const Shape<2>& shape() const { return *this->shape_; }
  inline size_t rows() const { return (*this->shape_)[0]; }
  inline size_t cols() const { return (*this->shape_)[1]; }

  inline size_t dim() const { return 2; }
  inline size_t dim(int idx) const { return (*this->shape_)[idx]; }

  inline bool empty() const {
    return this->shape_ == nullptr || this->size() == 0;
  }

  /// accessing elements
  inline DType* data(size_t y = 0) {
    return this->storage_->begin() + y * this->shape_->offset(0);
  }
  inline const DType* data(size_t y = 0) const {
    return this->storage_->begin() + y * this->shape_->offset(0);
  }

  inline DType& operator()(size_t x, size_t y) { return this->data(y)[x]; }

  inline const DType& operator()(size_t x, size_t y) const {
    return this->data()[x];
  }

  inline DType& operator[](size_t idx) { return this->storage_->begin()[idx]; }
  inline const DType& operator[](size_t idx) const {
    return this->storage_->begin()[idx];
  }

  template <typename DType2>
  friend std::ostream& operator<<(std::ostream& os, const Matrix<DType2>& mat);

  template <typename DType2>
  friend std::istream& operator>>(std::istream& is, Matrix<DType2>& mat);

 protected:
  MatrixStorage<DType>* storage_;
  Shape<2>* shape_;
  int* count_;
};

template <typename DType>
std::ostream& operator<<(std::ostream& os, const Matrix<DType>& mat) {
  if (mat.empty()) {
    os << "[]";
  } else {
    const Shape<2>& s2d = *(mat.shape_);
    const DType* pdata = mat.data();
    for (size_t i = 0; i < s2d[0]; ++i) {
      os << "\n[ ";
      for (size_t j = 0; j < s2d[1]; ++j) os << *pdata++ << " ";
      os << "]";
    }
  }
  return os;
}

template <typename DType>
std::istream& operator>>(std::istream& is, Matrix<DType>& mat) {
  const Shape<2>& s2d = *(mat.shape_);
  DType* pdata = mat.data();
  for (size_t i = 0; i < s2d[0]; ++i) {
    is.ignore((std::numeric_limits<std::streamsize>::max)(), '[');
    for (size_t j = 0; j < s2d[1]; ++j) is >> *pdata++;
    is.ignore((std::numeric_limits<std::streamsize>::max)(), ']');
    is.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');
  }
  return is;
}

}  // namespace math
}  // namespace sol

#endif
