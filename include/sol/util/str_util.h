/*********************************************************************************
*     File Name           :     str_util.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 15:55]
*     Last Modified       :     [2017-05-09 15:37]
*     Description         :      string related operations
**********************************************************************************/
#ifndef SOL_UTIL_STR_UTIL_H__
#define SOL_UTIL_STR_UTIL_H__

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace sol {

inline std::vector<std::string> split(const std::string& str,
                                      char delim = '\t') {
  std::vector<std::string> res;
  auto e = str.end();
  auto i = str.begin();
  while (i != e) {
    i = std::find_if_not(i, e, [delim](char c) { return c == delim; });
    if (i == e) break;
    auto j = find_if(i, e, [delim](char c) { return c == delim; });
    res.push_back(std::string(i, j));
    i = j;
  }
  return res;
}

inline std::string strip(const std::string& str) {
  auto i = std::find_if_not(str.begin(), str.end(),
                            [](char c) { return c == ' ' || c == '\t'; });
  auto j =
      std::find_if(i, str.end(), [](char c) { return c == ' ' || c == '\t'; });
  return std::string(i, j);
}

inline std::string lower(const std::string& str) {
  std::string res = str;
  for (char& c : res) {
    c = tolower(c);
  }
  return res;
}

/// \brief  hash functions of string
namespace strhash {
template <class>
struct hasher;
template <>
struct hasher<std::string> {
  std::size_t constexpr operator()(char const* input) const {
    return *input ? static_cast<unsigned int>(*input) + 33 * (*this)(input + 1)
                  : 5381;
  }
  inline std::size_t operator()(const std::string& str) const {
    return (*this)(str.c_str());
  }
};
}

template <typename T>
std::size_t constexpr str2int(T&& t) {
  return strhash::hasher<typename std::decay<T>::type>()(std::forward<T>(t));
}

// convert string to integer
inline std::size_t constexpr operator"" _I(const char* s, size_t) {
  return strhash::hasher<std::string>()(s);
}

}  // namespace std

#endif
