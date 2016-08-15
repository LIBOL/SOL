/*********************************************************************************
*     File Name           :     platform_win32.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-23 13:33]
*     Last Modified       :     [2015-12-03 14:49]
*     Description         :     platform specific functions for windows
**********************************************************************************/

#ifndef LSOL_UTIL_PLATFORM_WIN32_H__
#define LSOL_UTIL_PLATFORM_WIN32_H__

#include <windows.h>

namespace lsol {

inline FILE* open_file(const char* path, const char* mode) {
  FILE* file;
  errno_t ret = fopen_s(&file, path, mode);
  if (ret != 0) {
    return nullptr;
  }
  return file;
}

}  // namespace lsol

#endif  // LSOL_UTIL_PLATFORM_WIN32_H__
