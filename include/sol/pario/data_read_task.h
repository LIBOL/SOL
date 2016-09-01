/*********************************************************************************
*     File Name           :     data_read_task.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-12 16:15]
*     Last Modified       :     [2016-02-12 18:07]
*     Description         :
**********************************************************************************/
#ifndef SOL_PARIO_DATA_READ_TASK_H__
#define SOL_PARIO_DATA_READ_TASK_H__

#include <sol/pario/data_point.h>
#include <sol/pario/mini_batch.h>
#include <sol/pario/data_reader.h>
#include <sol/util/block_queue.h>
#include <sol/util/thread_task.h>

namespace sol {
namespace pario {

/// \brief  Thread task to read data
class DataReadTask : public ThreadTask {
 public:
  /// \brief  Initialize the Data Read Task
  ///
  /// \param path data file path
  /// \param dtype data type (svm, bin, csv, etc.)
  /// \param mini_batch_factory factory of empty mini batch
  /// \param mini_batch_buf place to store the loaded mini batched
  /// \param pass_num number of passes to read the data
  DataReadTask(const std::string& path, const std::string& dtype,
               BlockQueue<MiniBatch*>& mini_batch_factory,
               BlockQueue<MiniBatch*>& mini_batch_buf, int pass_num);

 public:
  inline bool Good() { return this->reader_ != nullptr; }

 protected:
  virtual void run();

 private:
  std::unique_ptr<DataReader> reader_;
  BlockQueue<MiniBatch*>& mini_batch_factory_;
  BlockQueue<MiniBatch*>& mini_batch_buf_;
  int pass_num_;
};

}  // namespace pario
}  // namespace sol
#endif
