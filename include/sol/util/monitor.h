/*********************************************************************************
*     File Name           :     monitor.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-05-14 15:18]
*     Last Modified       :     [2016-05-17 02:30]
*     Description         :
**********************************************************************************/

#ifndef CXX_SELF_CUSTOMIZED_MONITOR_H__
#define CXX_SELF_CUSTOMIZED_MONITOR_H__

#include <sol/util/mutex.h>

#if USE_STD_THREAD
#include <condition_variable>
#endif

#include <memory>

namespace sol {

#if USE_STD_THREAD
class Monitor {
 public:
  Monitor() : owned_mutex_(new Mutex()) {
    this->mutex_ = this->owned_mutex_.get();
  }
  Monitor(Mutex& mutex) { this->mutex_ = &mutex; }

  void lock() { this->mutex_->lock(); }
  void unlock() { this->mutex_->unlock(); }
  // assume the thread already obtains the lock
  void wait() {
    std::unique_lock<std::mutex> lock(this->mutex_->mutex(), std::adopt_lock);
    this->cv_.wait(lock);
    lock.release();
  }
  void notify() { this->cv_.notify_one(); }
  void notify_all() { this->cv_.notify_all(); }

 protected:
  std::unique_ptr<Mutex> owned_mutex_;
  std::condition_variable cv_;
  Mutex* mutex_;
};

#elif USE_WIN_THREAD
class Monitor {
 public:
  Monitor() : owned_mutex_(new Mutex()) {
    this->mutex_ = this->owned_mutex_.get();
    InitializeConditionVariable(&cv_);
  }
  Monitor(Mutex& mutex) {
    this->mutex_ = &mutex;

    InitializeConditionVariable(&cv_);
  }

  void lock() { this->mutex_->lock(); }
  void unlock() { this->mutex_->unlock(); }
  // assume the thread already obtains the lock
  void wait() {
    SleepConditionVariableCS(&cv_, &(this->mutex_->mutex()), INFINITE);
  }
  void notify() { WakeConditionVariable(&cv_); }
  void notify_all() { WakeAllConditionVariable(&cv_); }

 protected:
  std::unique_ptr<Mutex> owned_mutex_;
  CONDITION_VARIABLE cv_;
  Mutex* mutex_;
};

#elif USE_PTHREAD

class Monitor {
 public:
  Monitor() : owned_mutex_(new Mutex()) {
    this->mutex_ = this->owned_mutex_.get();
    pthread_cond_init(&cv_, nullptr);
  }
  Monitor(Mutex& mutex) {
    this->mutex_ = &mutex;
    pthread_cond_init(&cv_, nullptr);
  }

  void lock() { this->mutex_->lock(); }
  void unlock() { this->mutex_->unlock(); }
  // assume the thread already obtains the lock
  void wait() { pthread_cond_wait(&cv_, &(this->mutex_->mutex())); }
  void notify() { pthread_cond_signal(&cv_); }
  void notify_all() { pthread_cond_broadcast(&cv_); }

 protected:
  std::unique_ptr<Mutex> owned_mutex_;
  pthread_cond_t cv_;
  Mutex* mutex_;
};

#endif

}  // namespace sol

#endif
