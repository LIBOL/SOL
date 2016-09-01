/*********************************************************************************
*     File Name           :     block_queue.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-20 20:18]
*     Last Modified       :     [2016-05-15 16:28]
*     Description         :     A thread-safe circular queue
* **********************************************************************************/

#ifndef CXX_SELF_CUSTOMIZED_BLOCK_QUEUE_H__
#define CXX_SELF_CUSTOMIZED_BLOCK_QUEUE_H__

#include <sol/util/mutex.h>
#include <sol/util/monitor.h>

namespace sol {

enum class QueueType {
  Element = 0,
  Array = 1,
  Managed = 2,
};

template <QueueType QType>
struct QueueType_traits;

template <>
struct QueueType_traits<QueueType::Element> {
  template <typename T>
  void operator()(T& elem) {
    if (elem != nullptr) {
      delete elem;
      elem = nullptr;
    }
  }
};

template <>
struct QueueType_traits<QueueType::Array> {
  template <typename T>
  void operator()(T& elem) {
    if (elem != nullptr) {
      delete[] elem;
      elem = nullptr;
    }
  }
};

template <>
struct QueueType_traits<QueueType::Managed> {
  template <typename T>
  void operator()(T& elem) {}
};

/// \brief  A fixed-size thread-safe circular queue
///
/// \tparam T Queue element type
template <typename T, QueueType QType = QueueType::Managed>
class BlockQueue {
 public:
  /// \brief a block queue to schedule tasks
  ///
  /// \param queue_size size of the task queue
  BlockQueue(int queue_size)
      : elems_(nullptr),
        elem_num_(0),
        head_(0),
        tail_(0),
        queue_size_(queue_size),
        lock_(),
        nonfull_(lock_),
        nonempty_(lock_) {
    this->elems_ = new T[queue_size];
  }
  ~BlockQueue() {
    QueueType_traits<QType> destory;
    while (this->elem_num_ > 0) {
      destory(this->elems_[this->head_]);
      this->head_ = (this->head_ + 1) % this->queue_size_;
      --this->elem_num_;
    }
    delete[] this->elems_;
  }

 public:
  /// \brief  Block until not full
  ///
  /// \param elem element to be enqueued
  void Enqueue(const T& elem) {
    this->lock_.lock();
    while (this->elem_num_ == this->queue_size_) {
      this->nonfull_.wait();
    }
    this->elems_[this->tail_] = elem;
    this->tail_ = (this->tail_ + 1) % this->queue_size_;
    ++this->elem_num_;
    this->nonempty_.notify();
    this->lock_.unlock();
  }

  /// \brief  enqueue a list of elements
  ///
  /// \param elem_list list of elements
  /// \param num number of elements
  void Enqueue(T* elem_list, int num) {
    this->lock_.lock();
    for (int i = 0; i < num; ++i) {
      while (this->elem_num_ == this->queue_size_) {
        this->nonempty_.notify_all();
        this->nonfull_.wait();
      }
      this->elems_[this->tail_] = elem_list[i];
      this->tail_ = (this->tail_ + 1) % this->queue_size_;
      ++this->elem_num_;
    }
    this->nonempty_.notify_all();
    this->lock_.unlock();
  }

  /// \brief  Blocks until not empty.
  T Dequeue() {
    this->lock_.lock();
    while (this->elem_num_ == 0) {
      this->nonempty_.wait();
    }
    T el = this->elems_[this->head_];
    this->head_ = (this->head_ + 1) % this->queue_size_;
    --this->elem_num_;
    this->nonfull_.notify_all();
    this->lock_.unlock();
    return el;
  }

  /// \brief  dequeeu a list of elements
  ///
  /// \param elem_list storage to save the elements
  /// \param num number of elements to dequeue
  /// \param force force to dequeue the number of elements
  ///
  /// \return number of elements that have been dequeued
  int Dequeue(T* elem_list, int num, bool force = false) {
    this->lock_.lock();
    // wait elements
    while (this->elem_num_ == 0) {
      this->nonempty_.wait();
    }

    int i = 0;
    for (i = 0; i < num; ++i) {
      if (this->elem_num_ == 0) {
        if (force == true && i > 0 && elem_list[i - 1] != nullptr) {
          this->nonfull_.notify_all();
          this->nonempty_.wait();
        } else {
          break;
        }
      }
      elem_list[i] = this->elems_[this->head_];
      this->head_ = (this->head_ + 1) % this->queue_size_;
      --this->elem_num_;
    }
    this->nonfull_.notify_all();
    this->lock_.unlock();
    return i;
  }

 public:
  int capacity() const { return this->queue_size_; }
  bool empty() const { return this->elem_num_ == 0; }
  int size() const { return this->elem_num_; }

 protected:
  T* elems_;
  int elem_num_;
  int head_, tail_;
  int queue_size_;
  Mutex lock_;
  Monitor nonfull_;
  Monitor nonempty_;
};

}  // namespace sol
#endif
