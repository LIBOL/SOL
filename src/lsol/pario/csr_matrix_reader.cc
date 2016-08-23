/*********************************************************************************
*     File Name           :     ../../../src/lsol/pario/csr_matrix_reader.cc
*     Created By          :     yuewu
*     Description         :     reader for csr_matrix
**********************************************************************************/

#ifdef HAS_NUMPY_DEV

#include "lsol/pario/csr_matrix_reader.h"

#include <cstdlib>
#include <sstream>

#include "lsol/util/str_util.h"

using namespace std;

namespace lsol {
namespace pario {

int CsrMatrixReader::Open(const std::string& path, const char* mode) {
  int ret = this->ParsePath(path, this->indices_, this->indptr_,
                            this->features_, this->Y_, this->n_samples_);
  this->is_good_ = ret == Status_OK ? true : false;
  this->x_idx_ = 0;
  return ret;
}

void CsrMatrixReader::Rewind() { this->x_idx_ = 0; }

std::string CsrMatrixReader::GeneratePath(int* indices, int* indptr,
                                          double* features, double* y,
                                          int n_samples) {
  ostringstream path;
  path << (long long)(indices) << ";" << (long long)(indptr) << ";"
       << (long long)(features) << ";" << (long long)(y) << ";" << n_samples;
  return path.str();
}

template <typename T>
int ParseAddr(T& dst, const std::string& src) {
  string::size_type sz = 0;
  try {
    string tmp = strip(src);
    dst = (T)(stoll(tmp, &sz));
    if (tmp[sz] != '\0') {
      return Status_IO_Error;
    }
  }
  catch (invalid_argument& err) {
    return Status_IO_Error;
  }
  return Status_OK;
}

int CsrMatrixReader::ParsePath(const std::string& path, int*& indices,
                               int*& indptr, double*& features, double*& y,
                               int& n_samples) {
  const vector<string>& parts = split(path, ';');
  if (parts.size() != 5) {
    fprintf(stderr, "invalid address (%s) for numpy reader\n", path.c_str());
    return Status_IO_Error;
  }
  int ret = Status_OK;
  ret = ParseAddr<int*>(indices, parts[0]);
  if (ret != Status_OK) return ret;
  ret = ParseAddr<int*>(indptr, parts[1]);
  if (ret != Status_OK) return ret;
  ret = ParseAddr<double*>(features, parts[2]);
  if (ret != Status_OK) return ret;
  ret = ParseAddr<double*>(y, parts[3]);
  if (ret != Status_OK) return ret;
  ret = ParseAddr<int>(n_samples, parts[4]);
  return ret;
}

int CsrMatrixReader::Next(DataPoint& dst_data) {
  if (this->x_idx_ == this->n_samples_) {
    return this->n_samples_ > 0 ? Status_EndOfFile : Status_IO_Error;
  }

  dst_data.Clear();
  // 1. parse label
  if (this->Y_ != nullptr) {
    dst_data.set_label(label_t(this->Y_[this->x_idx_]));
  }

  // 2. parse features
  int feat_num = this->indptr_[this->x_idx_ + 1] - this->indptr_[this->x_idx_];
  int offset = this->indptr_[this->x_idx_] - this->indptr_[0];
  double* feat_ptr = this->features_ + offset;
  int* indice_ptr = this->indices_ + offset;

  for (int j = 0; j < feat_num; ++j, ++feat_ptr, ++indice_ptr) {
    dst_data.AddNewFeat(*indice_ptr + 1, *feat_ptr);
  }
  dst_data.Sort();
  ++this->x_idx_;
  return Status_OK;
}

RegisterDataReader(CsrMatrixReader, "csr_matrix",
                   "csr_matrix array data reader");

}  // namespace pario
}  // namespace lsol

#endif
