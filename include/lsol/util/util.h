/*********************************************************************************
*     File Name           :     platform.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-17 11:06]
*     Last Modified       :     [2016-02-12 17:51]
*     Description         :     utilized or platform specific functions
**********************************************************************************/

#ifndef LSOL_UTIL_PLATFORM_H__
#define LSOL_UTIL_PLATFORM_H__

#include <cstdio>
#include <cstdlib>

/// \brief  declaration of functions
namespace lsol {

#define DeletePointer(p)  \
  if ((p) != nullptr) { \
    delete (p);         \
    (p) = nullptr;      \
  }

#define DeleteArray(p)    \
  if ((p) != nullptr) { \
    delete[](p);        \
    (p) = nullptr;      \
  }

#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&);             \
  classname& operator=(const classname&);

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

}  // namespace lsol

#if _WIN32
#include "lsol/util/platform_win32.h"
#else
#include "lsol/util/platform_xnix.h"
#endif

#endif
