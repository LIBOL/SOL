/*********************************************************************************
*     File Name           :     data_iter.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-12 15:00]
*     Last Modified       :     [2016-02-12 18:26]
*     Description         :     Data Iterator
**********************************************************************************/

#include "lsol/pario/data_iter.h"
#include "lsol/util/util.h"

using namespace std;

namespace lsol {
namespace pario {
DataIter::DataIter(int batch_size, int batch_num)
    : batch_size_(batch_size),
      mini_batch_factory_(batch_num),
      mini_batch_buf_(batch_num) {
  for (int i = 0; i < batch_num; ++i) {
    this->mini_batch_factory_.Enqueue(new MiniBatch(batch_size));
  }
  this->running_readers_ = 0;
}

DataIter::~DataIter() {
  MiniBatch* mb = nullptr;
  // clear mini_batch_factory_
  while (this->mini_batch_factory_.size() > 0) {
    mb = this->mini_batch_factory_.Dequeue();
    DeletePointer(mb);
  }

  //send exit signal to readers
  this->mini_batch_factory_.Enqueue(mb);

  // clear mini_batch_buf_
  while (this->running_readers_ > 0) {
	  mb = this->mini_batch_buf_.Dequeue();
	  if (mb == nullptr) {
		  --this->running_readers_;
	  }
	  else {
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
    ++this->running_readers_;
    reader->Start();
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
      --this->running_readers_;
    }
  } while (el == nullptr && this->running_readers_ > 0);
  return el;
}

}  // namespace pario
}  // namespace lsol
