/*********************************************************************************
*     File Name           :     platform_xnix.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-23 13:37]
*     Last Modified       :     [2015-12-03 14:49]
*     Description         :     platform specific functions for linix/unix
**********************************************************************************/

#ifndef SOL_UTIL_PLATFORM_XNIX_H__
#define SOL_UTIL_PLATFORM_XNIX_H__

namespace sol {

inline FILE* open_file(const char* path, const char* mode) {
  return fopen(path, mode);
}

}  // namespace sol

#endif
