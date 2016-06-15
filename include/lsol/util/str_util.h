/*********************************************************************************
*     File Name           :     str_util.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 15:55]
*     Last Modified       :     [2016-02-18 23:12]
*     Description         :      string related operations
**********************************************************************************/

#include <string>
#include <vector>
#include <algorithm>
#include <utility>

namespace std {

std::vector<std::string> split(const std::string& str, char delim = '\t') {
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

std::string strip(const std::string& str) {
  auto i =
      std::find_if_not(str.begin(), str.end(), [](char c) { return c == ' '; });
  auto j = std::find_if(i, str.end(), [](char c) { return c == ' '; });
  return std::move(std::string(i, j));
}

}  // namespace std
