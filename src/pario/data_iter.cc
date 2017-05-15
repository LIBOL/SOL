/*********************************************************************************
*     File Name           :     data_iter.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-12 15:00]
*     Last Modified       :     [2016-02-12 18:26]
*     Description         :     Data Iterator
**********************************************************************************/

#include "sol/pario/data_iter.h"
#include "sol/util/util.h"

using namespace std;

namespace sol {
namespace pario {
DataIter::DataIter(int batch_size, int batch_num)
    : batch_size_(batch_size),
      mini_batch_factory_(batch_num),
      mini_batch_buf_(batch_num) {
  for (int i = 0; i < batch_num; ++i) {
    this->mini_batch_factory_.Enqueue(new MiniBatch(batch_size));
  }
  // signal to start next reader
  this->mini_batch_buf_.Enqueue(nullptr);
  this->running_reader_idx_ = -1;
}

DataIter::~DataIter() {
  MiniBatch* mb = nullptr;
  // clear mini_batch_factory_
  while (this->mini_batch_factory_.size() > 0) {
    mb = this->mini_batch_factory_.Dequeue();
    DeletePointer(mb);
  }

  // send exit signal to readers
  this->mini_batch_factory_.Enqueue(mb);

  // clear mini_batch_buf_
  int reader_count = static_cast<int>(this->readers_.size());
  while (this->running_reader_idx_ < reader_count) {
    mb = this->mini_batch_buf_.Dequeue();
    if (mb == nullptr) {
      ++this->running_reader_idx_;
      if (this->running_reader_idx_ < reader_count) {
        this->readers_[this->running_reader_idx_]->Start();
      }
    } else {
      DeletePointer(mb);
    }
  }

  // wait all data readers to exit
  for (shared_ptr<DataReadTask>& reader : this->readers_) {
    reader->Join();
  }

  // clear mini_batch_factory_
  while (this->mini_batch_factory_.size() > 0) {
    mb = this->mini_batch_factory_.Dequeue();
    DeletePointer(mb);
  }
}

int DataIter::AddReader(const std::string& path, const std::string& dtype,
                        int pass_num) {
  int ret = Status_OK;
  shared_ptr<DataReadTask> reader(new DataReadTask(
      path, dtype, this->mini_batch_factory_, this->mini_batch_buf_, pass_num));
  if (reader->Good()) {
    this->readers_.push_back(reader);
  } else {
    ret = Status_Invalid_Argument;
    fprintf(stderr, "add reader (type: %s, path: %s) failed\n", dtype.c_str(),
            path.c_str());
  }
  return ret;
}

MiniBatch* DataIter::Next(MiniBatch* prev_batch) {
  if (prev_batch != nullptr) {
    this->mini_batch_factory_.Enqueue(prev_batch);
  }
  MiniBatch* el = nullptr;
  do {
    el = this->mini_batch_buf_.Dequeue();
    if (el == nullptr) {
      if (this->running_reader_idx_ + 1 <
          static_cast<int>(this->readers_.size())) {
        ++this->running_reader_idx_;
        this->readers_[this->running_reader_idx_]->Start();
      } else {
        this->mini_batch_buf_.Enqueue(nullptr);
        break;
      }
    }
  } while (el == nullptr);
  return el;
}

}  // namespace pario
}  // namespace sol
