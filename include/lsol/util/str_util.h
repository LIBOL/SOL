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

vector<string> split(const string& str, char delim = '\t') {
  vector<string> res;
  auto e = str.end();
  auto i = str.begin();
  while (i != e) {
    i = find_if_not(i, e, [delim](char c) { return c == delim; });
    if (i == e) break;
    auto j = find_if(i, e, [delim](char c) { return c == delim; });
    res.push_back(string(i, j));
    i = j;
  }
  return std::move(res);
}

}  // namespace std
