/*********************************************************************************
*     File Name           :     numpy_reader.h
*     Created By          :     yuewu
*     Description         :     reader for numpy array
**********************************************************************************/

#ifdef HAS_NUMPY_DEV
#ifndef LSOL_PARIO_NUMPY_READER_H__
#define LSOL_PARIO_NUMPY_READER_H__

#include <lsol/pario/data_reader.h>
#include <numpy/arrayobject.h>

namespace lsol {
namespace pario {

class LSOL_EXPORTS NumpyReader : public DataReader {
 public:
  NumpyReader()
      : is_good_(true),
        X_(nullptr),
        Y_(nullptr),
        n_samples_(0),
        n_features_(0),
        stride_(0),
        x_idx_(0) {}
  virtual ~NumpyReader() {}

 public:
  virtual int Open(const std::string& path, const char* mode = "r");
  virtual void Close() {}

  virtual bool Good() { return is_good_; }

  virtual void Rewind();

 public:
  virtual int Next(DataPoint& dst_data);

 public:
  static std::string GeneratePath(double* x, double* y, int rows, int cols,
                                  int stride);
  static int ParsePath(const std::string& path, double*& x, double*& y,
                       int& rows, int& cols, int& stride);

 protected:
  bool is_good_;
  double* X_;
  double* Y_;
  int n_samples_;
  int n_features_;
  int stride_;
  int x_idx_;
};

}  // namespace pario
}  // namespace lsol

#endif
#endif
