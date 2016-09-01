/*********************************************************************************
*     File Name           :     mutex.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-05-14 15:09]
*     Last Modified       :     [2016-05-14 23:29]
*     Description         :
**********************************************************************************/

#ifndef CXX_SELF_CUSTOMIZED_MUTEX_H__
#define CXX_SELF_CUSTOMIZED_MUTEX_H__

#if USE_WIN_THREAD
#include <windows.h>
#elif USE_PTHREAD
#include <pthread.h>
#else
#include <mutex>
#ifndef USE_STD_THREAD
#define USE_STD_THREAD 1
#endif
#endif

namespace sol {

#if USE_STD_THREAD
class Mutex {
 public:
  Mutex() {}

  void lock() { this->mutex_.lock(); }
  void unlock() { this->mutex_.unlock(); }

  std::mutex& mutex() { return mutex_; }

 protected:
  std::mutex mutex_;
};

#elif USE_WIN_THREAD
class Mutex {
 public:
  Mutex() { InitializeCriticalSection(&mutex_); }
  ~Mutex() { DeleteCriticalSection(&mutex_); }

  void lock() { EnterCriticalSection(&mutex_); }
  void unlock() { LeaveCriticalSection(&mutex_); }

  CRITICAL_SECTION& mutex() { return mutex_; }

 protected:
  CRITICAL_SECTION mutex_;
};

#elif USE_PTHREAD

class Mutex {
 public:
  Mutex() { pthread_mutex_init(&mutex_, nullptr); }

  void lock() { pthread_mutex_lock(&mutex_); }
  void unlock() { pthread_mutex_unlock(&mutex_); }

  pthread_mutex_t& mutex() { return mutex_; }

 protected:
  pthread_mutex_t mutex_;
};

#endif

}  // namespace sol

#endif
