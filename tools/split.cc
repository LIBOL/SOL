/*********************************************************************************
*     File Name           :     shuffle.cc
*     Created By          :     yuewu
*     Description         :     split file into folds
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
  parser.add<int>("fold", 'n', "split number", true);
  parser.add<string>("output_prefix", 'o', "output prefix", true);
  parser.add<string>("output_type", 'd', "output data type");
  parser.add("shuffle", 'r', "shuffle the input file");

  parser.parse_check(argc, argv);

  return split(parser.get<string>("input"), parser.get<string>("input_type"),
               parser.get<int>("fold"), parser.get<string>("output_prefix"),
               parser.get<string>("output_type"),
               parser.exist("shuffle") ? true : false);
}
