/*********************************************************************************
*     File Name           :     platform.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-17 11:06]
*     Last Modified       :     [2016-02-12 17:51]
*     Description         :     utilized or platform specific functions
**********************************************************************************/

#ifndef LSOL_UTIL_PLATFORM_H__
#define LSOL_UTIL_PLATFORM_H__

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

/// \brief  declaration of functions
namespace lsol {

#define DeletePointer(p) \
  if ((p) != nullptr) {  \
    delete (p);          \
    (p) = nullptr;       \
  }

#define DeleteArray(p)  \
  if ((p) != nullptr) { \
    delete[](p);        \
    (p) = nullptr;      \
  }

#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&);             \
  classname& operator=(const classname&);

// check if the argument is valid and throw exception otherwise
#define Check(condition)                                                      \
  if ((condition) == false) {                                                 \
    std::ostringstream oss;                                                   \
    oss << "Check " << #condition << " failed at line " << __LINE__ << " of " \
        << __FILE__;                                                          \
    throw std::invalid_argument(oss.str());                                   \
  }

/// \brief  Open file wrapper, windows use fopen_s for safety
///
/// \param path Path to file
/// \param mode Open mode
///
/// \return FILE pointer
inline FILE* open_file(const char* path, const char* mode);

/// \brief  delete a file
///
/// \param path File path
/// \param is_force Whether prompt when file not exist
inline void delete_file(const char* path, bool is_force = false) {
  if (remove(path) != 0 && is_force == false) {
    fprintf(stderr, "warnning, remove file %s failed!\n", path);
  }
}

/// \brief  get current time, in seconds
///
/// \return seconds
inline double get_current_time() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
             .count() *
         0.001;
}

}  // namespace lsol

#if _WIN32
#include "lsol/util/platform_win32.h"
#else
#include "lsol/util/platform_xnix.h"
#endif

#endif
