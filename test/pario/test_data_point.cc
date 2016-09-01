/*********************************************************************************
*     File Name           :     test_data_point.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-29 13:52]
*     Last Modified       :     [2015-11-12 18:11]
*     Description         :
**********************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <sol/pario/data_point.h>

using namespace sol;
using namespace sol::pario;
using namespace std;

int main() {
// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(tmpFlag);
//_CrtSetBreakAlloc(368);
#endif

  size_t N = 4;

  DataPoint pt;

  for (size_t i = 0; i < N; ++i) {
    index_t idx = rand() % N + 1;
    real_t feat = (real_t)(rand() % N);
    cout << "add new feat: " << idx << ": " << feat << endl;
    pt.AddNewFeat(idx, feat);
  }
  cout << "original data point" << endl;
  for (size_t i = 0; i < pt.size(); ++i) {
    cout << pt.indexes()[i] << ": " << pt.features()[i] << endl;
  }

  cout << "sort" << endl;
  pt.Sort();
  for (size_t i = 0; i < pt.size(); ++i) {
    cout << pt.indexes()[i] << ": " << pt.features()[i] << endl;
  }

  return 0;
}
