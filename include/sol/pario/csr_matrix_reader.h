/*********************************************************************************
*     File Name           :     csr_matrix_reader.h
*     Created By          :     yuewu
*     Description         :     reader for csr_matrix
**********************************************************************************/

#ifdef HAS_NUMPY_DEV
#ifndef SOL_PARIO_CSR_MATRIX_READER_H__
#define SOL_PARIO_CSR_MATRIX_READER_H__

#include <sol/pario/data_reader.h>
#include <numpy/arrayobject.h>

namespace sol {
namespace pario {

class SOL_EXPORTS CsrMatrixReader : public DataReader {
 public:
  CsrMatrixReader()
      : is_good_(true),
        indices_(nullptr),
        indptr_(nullptr),
        features_(nullptr),
        Y_(nullptr),
        n_samples_(0),
        x_idx_(0) {}
  virtual ~CsrMatrixReader() {}

 public:
  virtual int Open(const std::string& path, const char* mode = "r");
  virtual void Close() {}

  virtual bool Good() { return is_good_; }

  virtual void Rewind();

 public:
  virtual int Next(DataPoint& dst_data);

 public:
  static std::string GeneratePath(int* indices, int* indptr, double* features,
                                  double* y, int n_samples);
  static int ParsePath(const std::string& path, int*& indices, int*& indptr,
                       double*& features, double*& y, int& n_samples);

 protected:
  bool is_good_;

  int* indices_;
  int* indptr_;
  double* features_;
  double* Y_;
  int n_samples_;
  int x_idx_;
};

}  // namespace pario
}  // namespace sol

#endif
#endif
