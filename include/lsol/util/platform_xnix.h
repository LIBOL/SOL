/*********************************************************************************
*     File Name           :     platform_xnix.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-23 13:37]
*     Last Modified       :     [2015-12-03 14:49]
*     Description         :     platform specific functions for linix/unix
**********************************************************************************/

#ifndef LSOL_UTIL_PLATFORM_XNIX_H__
#define LSOL_UTIL_PLATFORM_XNIX_H__

#include <sys/time.h>
#include <sys/types.h>

namespace lsol {

inline FILE* open_file(const char* path, const char* mode) {
  return fopen(path, mode);
}

inline double get_current_time() {
  struct timeval tim;
  gettimeofday(&tim, NULL);
  return tim.tv_sec + tim.tv_usec / 1000000.0;
}

}  // namespace lsol

#endif
