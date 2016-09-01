/*********************************************************************************
*     File Name           :     thread_task.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-12-03 14:15]
*     Last Modified       :     [2016-05-14 00:21]
*     Description         :     Task processed in a separate task
**********************************************************************************/

#ifndef SHENTU_UTIL_THREAD_TASK_H__
#define SHENTU_UTIL_THREAD_TASK_H__

#include <memory>
#include <sol/util/thread.h>

namespace sol {

/// \brief  task processed in a separate task
class ThreadTask {
 public:
  void Start() {
    if (this->thread_ == nullptr) {
      this->thread_.reset(new Thread(ThreadTask::InternalEntry, this));
    }
  }
  void Join() {
    if (this->thread_) {
      this->thread_->join();
    }
  }

 protected:
  virtual void run() = 0;
  static void InternalEntry(void* task) { ((ThreadTask*)task)->run(); }

 protected:
  std::unique_ptr<Thread> thread_;
};  // class ThreadTask

}  // namespace sol
#endif
