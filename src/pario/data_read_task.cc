/*********************************************************************************
*     File Name           :     data_read_task.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-12 15:18]
*     Last Modified       :     [2016-03-09 19:24]
*     Description         :
**********************************************************************************/

#include "sol/pario/data_read_task.h"
#include "sol/util/error_code.h"

namespace sol {
namespace pario {
DataReadTask::DataReadTask(const std::string& path, const std::string& dtype,
                           BlockQueue<MiniBatch*>& mini_batch_factory,
                           BlockQueue<MiniBatch*>& mini_batch_buf, int pass_num)
    : mini_batch_factory_(mini_batch_factory),
      mini_batch_buf_(mini_batch_buf),
      pass_num_(pass_num) {
  DataReader* reader = DataReader::Create(dtype);
  if (reader != nullptr) {
    if (reader->Open(path) != Status_OK) {
      delete reader;
      reader = nullptr;
    }
  }
  this->reader_.reset(reader);
}

void DataReadTask::run() {
  int status = Status_OK;
  DataReader* reader = this->reader_.get();
  while (status == Status_OK && this->pass_num_ > 0) {
    MiniBatch* mini_batch = this->mini_batch_factory_.Dequeue();
    if (mini_batch == nullptr) {  // exit signal
      this->mini_batch_factory_.Enqueue(nullptr);
      break;
    }
    mini_batch->data_num = 0;
    while (mini_batch->data_num < mini_batch->capacity() &&
           status == Status_OK) {
      status = reader->Next((*mini_batch)[mini_batch->data_num]);
      if (status == Status_OK) {
        ++mini_batch->data_num;
        continue;
      } else if (status == Status_EndOfFile) {
        --this->pass_num_;
        reader->Rewind();
        status = Status_OK;
        break;
      } else
        break;
    }
    this->mini_batch_buf_.Enqueue(mini_batch);
  }
  reader->Close();
  this->mini_batch_buf_.Enqueue(nullptr);
}

}  // namespace pario
}  // namespace sol
