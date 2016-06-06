/*********************************************************************************
*     File Name           :     test_compress.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 17:21]
*     Last Modified       :     [2015-11-14 17:48]
*     Description         :     test compression
**********************************************************************************/

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <time.h>

#include "lsol/pario/compress.h"

using namespace lsol::pario;
using namespace std;

template <typename T>
int test() {
  srand(static_cast<unsigned int>(time(0)));
  Array1d<T> arr;
  int N = 1000;
  for (int i = 0; i < N; ++i) {
    arr.Push(T(rand()));
  }
  std::sort(arr.begin(), arr.end());
  cout << "example array: " << endl;
  for (int i = 0; i < (N < 10 ? N : 10); ++i) {
    cout << arr[i] << " ";
  }
  cout << endl;

  Array1d<char> codes;
  comp_index(arr, codes);

  cout << "size of original array :" << sizeof(T) * N << " bytes" << endl;
  cout << "size of codes array    :" << sizeof(char) * codes.size() << " bytes"
       << endl;

  cout << "check decompress..." << endl;

  Array1d<T> arr2;
  decomp_index(codes, arr2);
  if (arr.size() != arr2.size()) {
    cerr << "size of decompressed array " << arr2.size() << " not equal to "
         << N << endl;
    return -1;
  }
  for (int i = 0; i < N; ++i) {
    if (arr[i] != arr2[i]) {
      cerr << i << "-th element not the same (" << arr[i] << " vs " << arr2[i]
           << ")" << endl;
      return -1;
    }
  }
  cout << "check decompress succeed" << endl;
  return 0;
}

int main() {
// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(tmpFlag);
//_CrtSetBreakAlloc(368);
#endif

  cout << "check uint16_t" << endl;
  int ret = test<uint16_t>();
  if (ret != 0) return ret;

  cout << "check uint32_t" << endl;
  ret = test<uint32_t>();
  if (ret != 0) return ret;

  cout << "check uint64_t" << endl;
  ret = test<uint64_t>();
  if (ret != 0) return ret;
  return 0;
}
