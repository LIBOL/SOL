/*********************************************************************************
*     File Name           :     str_util.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 15:55]
*     Last Modified       :     [2016-02-18 23:12]
*     Description         :      string related operations
**********************************************************************************/
#ifndef LSOL_UTIL_STR_UTIL_H__
#define LSOL_UTIL_STR_UTIL_H__

#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <cctype>

namespace lsol {

inline std::vector<std::string> split(const std::string& str,
                                      char delim = '\t') {
  std::vector<std::string> res;
  auto e = str.end();
  auto i = str.begin();
  while (i != e) {
    i = find_if_not(i, e, [delim](char c) { return c == delim; });
    if (i == e) break;
    auto j = find_if(i, e, [delim](char c) { return c == delim; });
    res.push_back(std::string(i, j));
    i = j;
  }
  return std::move(res);
}

inline std::string strip(const std::string& str) {
  auto i = std::find_if_not(str.begin(), str.end(),
                            [](char c) { return c == ' ' || c == '\t'; });
  auto j =
      std::find_if(i, str.end(), [](char c) { return c == ' ' || c == '\t'; });
  return std::move(std::string(i, j));
}

inline std::string lower(const std::string& str) {
  std::string res = str;
  for (char& c : res) {
    c = tolower(c);
  }
  return std::move(res);
}

}  // namespace std

#endif
