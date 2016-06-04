/*********************************************************************************
*     File Name           :     ../util/thread.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-05-14 00:22]
*     Last Modified       :     [2016-05-14 23:34]
*     Description         :
**********************************************************************************/

#ifndef CXX_SELF_CUSTOMIZED_THREAD_H__
#define CXX_SELF_CUSTOMIZED_THREAD_H__

#include <lsol/util/mutex.h>

#if USE_STD_THREAD
#include <thread>
#endif

namespace lsol {

typedef void (*ThreadFuncType)(void*);

#if USE_STD_THREAD
class Thread {
 public:
  Thread(ThreadFuncType start_routine, void* param) {
    this->thread_.reset(new std::thread(start_routine, param));
  }

  void join() { this->thread_->join(); }

 protected:
  std::unique_ptr<std::thread> thread_;
};

#elif USE_WIN_THREAD

class Thread {
 public:
  Thread(ThreadFuncType start_routine, void* param)
      : thread_func_(start_routine), thread_func_param_(param) {
    this->thread_ =
        CreateThread(nullptr, 0, Thread::ThreadProxy, this, 0, nullptr);
  }
  ~Thread() {
    TerminateThread(this->thread_, 0);
    CloseHandle(this->thread_);
  }

  void join() { WaitForSingleObject(this->thread_, INFINITE); }

 public:
  static DWORD WINAPI ThreadProxy(LPVOID param) {
    Thread* instance = (Thread*)(param);
    instance->thread_func_(instance->thread_func_param_);
    return 0;
  }

 protected:
  HANDLE thread_;
  ThreadFuncType thread_func_;
  void* thread_func_param_;
};

#elif USE_PTHREAD

class Thread {
 public:
  Thread(ThreadFuncType start_routine, void* param)
      : thread_func_(start_routine), thread_func_param_(param) {
    pthread_create(&thread_, nullptr, Thread::ThreadProxy, this);
  }

  void join() { pthread_join(this->thread_, nullptr); }

  static void* ThreadProxy(void* param) {
    Thread* instance = (Thread*)(param);
    instance->thread_func_(instance->thread_func_param_);
    return nullptr;
  }

 protected:
  pthread_t thread_;

  ThreadFuncType thread_func_;
  void* thread_func_param_;
};

#endif

}  // namespace lsol
#endif
