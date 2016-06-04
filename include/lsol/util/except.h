/*********************************************************************************
*     File Name           :     except.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-16 22:55]
*     Last Modified       :     [2016-02-18 17:10]
*     Description         :      definition of exceptions
**********************************************************************************/

#include <stdexcept>
#include <sstream>

// check if the argument is valid and throw exception otherwise
#define Check(condition)                                                      \
  if ((condition) == false) {                                                 \
    std::ostringstream oss;                                                   \
    oss << "Check " << #condition << " failed at line " << __LINE__ << " of " \
        << __FILE__;                                                          \
    throw std::invalid_argument(oss.str());                                   \
  }
