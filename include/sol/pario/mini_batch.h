/*********************************************************************************
*     File Name           :     mini_batch.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-12-03 16:59]
*     Last Modified       :     [2016-02-12 18:19]
*     Description         :     min batch
**********************************************************************************/

#ifndef SOL_PARIO_MINI_BATCH_H__
#define SOL_PARIO_MINI_BATCH_H__

#include <vector>

#include <sol/util/types.h>
#include <sol/pario/data_point.h>

namespace sol {
namespace pario {

class SOL_EXPORTS MiniBatch {
 public:
  MiniBatch(int batch_size = 0)
      : data_num(0), points_(nullptr), capacity_(batch_size) {
    this->points_ = new DataPoint[this->capacity_];
  }
  ~MiniBatch() {
    if (this->points_ != nullptr) {
      delete[] this->points_;
    }
  }

 public:
  inline const DataPoint* points() const { return this->points_; }
  inline int size() const { return this->data_num; }
  inline int capacity() const { return this->capacity_; }
  inline const DataPoint& operator[](size_t index) const {
    return this->points_[index];
  }
  inline DataPoint& operator[](size_t index) { return this->points_[index]; }

  int data_num;

 private:
  DataPoint* points_;
  int capacity_;
};

}  // namespace pario
}  // namespace sol
#endif
