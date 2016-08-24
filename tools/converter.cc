/*********************************************************************************
*     File Name           :     converter.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-12 18:29]
*     Last Modified       :     [2016-02-12 23:14]
*     Description         :     covert data formats
**********************************************************************************/

#include <lsol/tools.h>
#include <cmdline/cmdline.h>

using namespace lsol;
using namespace std;

int main(int argc, char** argv) {
// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(tmpFlag);
//_CrtSetBreakAlloc(231);
#endif

  cmdline::parser parser;
  parser.add<string>("input", 'i', "input data path", true);
  parser.add<string>("input_type", 's', "input data type", true);
  parser.add<string>("output", 'o', "output data path", true);
  parser.add<string>("output_type", 'd', "output data type", true);

  parser.parse_check(argc, argv);

  return convert(parser.get<string>("input"), parser.get<string>("input_type"),
                 parser.get<string>("output"),
                 parser.get<string>("output_type"));
}
