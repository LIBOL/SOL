/*********************************************************************************
*     File Name           :     numpy_reader.cc
*     Created By          :     yuewu
*     Description         :     reader for numpy array
**********************************************************************************/
#ifdef HAS_NUMPY_DEV

#include "lsol/pario/numpy_reader.h"

#include <cstdlib>
#include <sstream>

#include "lsol/util/str_util.h"

using namespace std;

namespace lsol {
namespace pario {

int NumpyReader::Open(const std::string& path, const char* mode) {
  int ret = this->ParsePath(path, this->X_, this->Y_, this->n_samples_,
                            this->n_features_, this->stride_);
  this->is_good_ = ret == Status_OK ? true : false;
  this->x_idx_ = 0;
  return ret;
}

void NumpyReader::Rewind() { this->x_idx_ = 0; }

std::string NumpyReader::GeneratePath(double* x, double* y, int rows, int cols,
                                      int stride) {
  ostringstream path;
  path << (long long)(x) << ";" << (long long)(y) << ";" << rows << ";" << cols
       << ";" << stride;
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
  catch (invalid_argument&) {
    return Status_IO_Error;
  }
  return Status_OK;
}

int NumpyReader::ParsePath(const std::string& path, double*& x, double*& y,
                           int& rows, int& cols, int& stride) {
  const vector<string>& parts = split(path, ';');
  if (parts.size() != 5) {
    fprintf(stderr, "invalid address (%s) for numpy reader\n", path.c_str());
    return Status_IO_Error;
  }
  int ret = Status_OK;
  ret = ParseAddr<double*>(x, parts[0]);
  if (ret != Status_OK) return ret;
  ret = ParseAddr<double*>(y, parts[1]);
  if (ret != Status_OK) return ret;
  ret = ParseAddr<int>(rows, parts[2]);
  if (ret != Status_OK) return ret;
  ret = ParseAddr<int>(cols, parts[3]);
  if (ret != Status_OK) return ret;
  ret = ParseAddr<int>(stride, parts[4]);
  return ret;
}

int NumpyReader::Next(DataPoint& dst_data) {
  if (this->x_idx_ == this->n_samples_)
    return this->n_samples_ > 0 ? Status_EndOfFile : Status_IO_Error;

  dst_data.Clear();
  // 1. parse label
  if (this->Y_ != nullptr) {
    dst_data.set_label(label_t(this->Y_[this->x_idx_]));
  }

  // 2. parse features
  double* ptr = (double*)((char*)this->X_ + this->x_idx_ * this->stride_);
  for (int j = 0; j < this->n_features_; ++j, ++ptr) {
    if (*ptr != 0) {
      dst_data.AddNewFeat(j + 1, static_cast<real_t>(*ptr));
    }
  }
  ++this->x_idx_;
  return Status_OK;
}

RegisterDataReader(NumpyReader, "numpy", "numpy array data reader");

}  // namespace pario
}  // namespace lsol

#endif
