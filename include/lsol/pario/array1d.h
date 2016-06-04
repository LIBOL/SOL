/*************************************************************************
  > File Name: Array1d.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/9/19 15:14:53
  > Functions: sparse array
 ************************************************************************/

#ifndef LSOL_PARIO_SARRAY_H__
#define LSOL_PARIO_SARRAY_H__

#include <cstring>
#include <stdexcept>
#include <cstdlib>

namespace lsol {
namespace pario {

// The difference of Array1d with vector is that vector copies the data, while
// Array1d only copies the pointer and increase counter
// Note that Array1d is not thread safe
template <typename T>
class Array1d {
 private:
  /// \brief  Memory component of Array1d
  template <typename DataType>
  struct Array1dMemory {
    // point to the first element
    DataType* begin;
    // point to the next postion of the last element
    DataType* end;
    // capacity of the array
    size_t capacity;

    Array1dMemory() : begin(nullptr), end(nullptr), capacity(0) {}

    Array1dMemory(const Array1dMemory<DataType>& mem)
        : begin(mem.begin), end(mem.end), capacity(mem.capacity) {}

    Array1dMemory<DataType>& operator=(const Array1dMemory<DataType>& mem) {
      this->begin = mem.begin;
      this->end = mem.end;
      this->capacity = mem.capacity;
    }
    ~Array1dMemory() {
      if (this->begin != nullptr) {
        delete[] this->begin;
      }
      this->begin = nullptr;
      this->end = nullptr;
      this->capacity = 0;
    }
  };

 public:
  Array1d() {
    this->sarray_mem_ = new Array1dMemory<T>;
    this->count_ = new int;
    *this->count_ = 1;
  }

  /// \brief  Specialized constructor from another array, note the constructed
  /// array will  point to the same memory space
  ///
  /// \param s_arr Source array
  Array1d(const Array1d<T>& s_arr)
      : sarray_mem_(s_arr.sarray_mem_), count_(s_arr.count_) {
    ++(*this->count_);
  }

  ~Array1d() {
    if (this->count_ != nullptr) {
      this->Release();
    }
  }

  /// \brief  assignment from Array1d, note the resulted array
  /// will point to the same place as source array
  ///
  /// \param exp The source array
  ///
  /// \return The assigned array (*this)
  Array1d<T>& operator=(const Array1d<T>& arr) {
    if (this->count_ != arr.count_) {
      this->Release();

      this->sarray_mem_ = arr.sarray_mem_;
      this->count_ = arr.count_;
      ++(*this->count_);
    }
    return *this;
  }

  /// \brief  Set the elements in the array to an arithmetic value
  ///
  /// \param val Value to be assigned
  inline void operator=(const T& val) {
    for (T* p = this->begin(); p < this->end(); ++p) *p = val;
  }

  /// \brief  Set the elements in the given range to the value
  ///
  /// \param iter_begin Beginning position of assignment
  /// \param iter_end End position of assignment
  /// \param val Value to be assigned
  void SetValue(T* iter_begin, T* iter_end, const T& val) {
    while (iter_begin < iter_end) {
      *iter_begin = val;
      iter_begin++;
    }
  }

  /// \brief  Reset all the elements in the array to zero
  void Zeros() { std::memset(this->begin(), 0, sizeof(T) * this->size()); }

  /// \brief  Set the elements in the given range of array to zero
  ///
  /// \param iter_begin Beginning position of assignment
  /// \param iter_end End position of assignment
  void Zeros(T* iter_begin, T* iter_end) {
    std::memset(iter_begin, 0, sizeof(T) * (iter_end - iter_begin));
  }

  /// \brief  Set all elements to 1
  void Ones() {
    size_t len = this->size();
    for (size_t i = 0; i < len; i++) this->begin()[i] = 1;
  }

 public:
  /// \brief  Resize the array to the given size, not the capacity is not
  /// ensured to be equal to the new size
  ///
  /// \param new_size New size to resize to
  void Resize(size_t new_size) {
    static size_t max_size = 1 << 30;

    if (this->capacity() < new_size) {
      // allocate more memory
      size_t alloc_size = this->capacity();
      do {
        alloc_size += (alloc_size < max_size ? alloc_size : max_size) + 3;
      } while (alloc_size < new_size);

      this->Allocate(alloc_size);
    }
    this->sarray_mem_->end = this->sarray_mem_->begin + new_size;
  }

  /// \brief  Reserver the specified space for array, the size of the array
  /// will be kept the same, but the capacity of the new array will exactly be
  /// the specified size
  ///
  /// \param new_size New size to reserve
  void Reserve(size_t new_size) {
    if (this->capacity() < new_size) {
      this->Allocate(new_size);
    }
  }

  /// \brief  Resize the array to be of size zero
  inline void Clear(void) { this->Resize(0); }

 public:
  /// \brief  Push a new element, Resize the array accordingly
  ///
  /// \param elem Element to be pushed
  inline void Push(const T& elem) {
    this->Resize(this->size() + 1);
    *(this->end() - 1) = elem;
  }

  /// \brief  Pop out and return the last element, work like a stack
  ///
  /// \return  The last element
  inline T& Pop() { return *(this->sarray_mem_->end--); }

 private:
  /// \brief  Release the spaces and counters
  void Release() {
    --(*this->count_);
    if (*this->count_ == 0) {
      delete this->sarray_mem_;
      this->sarray_mem_ = nullptr;
      delete this->count_;
    }
    this->count_ = nullptr;
  }

  /// \brief  Allocate the specified space for array
  ///
  /// \param new_size Specified size of memory to be allocated
  void Allocate(size_t new_size) {
    T* new_begin = nullptr;
    try {
      new_begin = new T[new_size];
    } catch (std::bad_alloc& ex) {
      fprintf(stderr, "%s\n", ex.what());
      fprintf(stderr,
              "realloc of %lu failed in resize(). out of memory ? in "
              "file %s line %d\n",
              new_size, __FILE__, __LINE__);
      exit(1);
    }
    if (new_begin == nullptr && sizeof(T) * new_size > 0) {
      fprintf(stderr,
              "realloc of %lu failed in resize(). out of memory ? in "
              "file %s line %d\n",
              new_size, __FILE__, __LINE__);
      exit(1);
    }

    size_t old_len = this->size();
    // copy data
    std::memcpy(new_begin, this->begin(), sizeof(T) * old_len);
    if (this->begin() != NULL) delete[] this->begin();
    this->sarray_mem_->begin = new_begin;
    this->sarray_mem_->end = new_begin + old_len;
    this->sarray_mem_->capacity = new_size;
  }

 public:
  inline const T* begin() const { return this->sarray_mem_->begin; }
  inline T* begin() { return this->sarray_mem_->begin; }
  inline const T& first() const { return *this->sarray_mem_->begin; }
  inline T& first() { return *this->sarray_mem_->begin; }

  inline const T* end() const { return this->sarray_mem_->end; }
  inline T* end() { return this->sarray_mem_->end; }
  inline const T& last() const { return *(this->sarray_mem_->end - 1); }
  inline T& last() { return *(this->sarray_mem_->end - 1); }

  inline T& operator[](size_t i) { return this->sarray_mem_->begin[i]; }
  inline const T& operator[](size_t i) const {
    return this->sarray_mem_->begin[i];
  }

  inline bool empty() const {
    return this->sarray_mem_->begin == this->sarray_mem_->end;
  }
  inline size_t size() const {
    return this->sarray_mem_->end - this->sarray_mem_->begin;
  }
  inline size_t dim() const {
    return this->sarray_mem_->end - this->sarray_mem_->begin;
  }
  inline size_t capacity() const { return this->sarray_mem_->capacity; }

 private:
  Array1dMemory<T>* sarray_mem_;
  int* count_;
};
}  // namespace pario
}  // namespace lsol

#endif  // LSOL_UTIL_SARRAY_H__
