/*********************************************************************************
*     File Name           :     data_iter.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-12-03 14:51]
*     Last Modified       :     [2016-02-12 18:10]
*     Description         :     Data Iterator
**********************************************************************************/

#ifndef LSOL_PARIO_DATA_ITER_H__
#define LSOL_PARIO_DATA_ITER_H__

#include <string>
#include <vector>
#include <memory>

#include "lsol/pario/data_point.h"
#include "lsol/pario/mini_batch.h"
#include "lsol/pario/data_reader.h"
#include "lsol/pario/data_read_task.h"
#include "lsol/util/block_queue.h"

namespace lsol {
namespace pario {

class LSOL_EXPORTS DataIter {
 public:
  /// \brief  Create a new Data Iterator
  ///
  /// \param batch_size size of minibatch
  /// \param batch_num number of mini-batches in buffer
  DataIter(int batch_size = 256, int batch_num = 2);
  virtual ~DataIter();

 public:
  /// \brief  Load a new data
  ///
  /// \param path data file path
  /// \param dtype data type (svm, bin, csv, etc.)
  ///
  /// \return
  int AddReader(const std::string& path, const std::string& dtype,
                int pass_num = 1);

  /// \brief  get the next mini-batch
  ///
  /// \param prev_batch previously used mini-batch for recycle
  ///
  /// \return
  virtual MiniBatch* Next(MiniBatch* prev_batch = nullptr);

 protected:
  // mini-batch size
  int batch_size_;
  // factory to store not used mini batches
  BlockQueue<MiniBatch*> mini_batch_factory_;
  // mini-batch number in buffer
  BlockQueue<MiniBatch*> mini_batch_buf_;
  // data reader threads
  std::vector<std::shared_ptr<DataReadTask>> readers_;
  // number of running readers
  int running_readers_;
};  // class DataIter
}  // namespace pario
}  // namespace lsol

#endif
