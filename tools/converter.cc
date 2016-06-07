/*********************************************************************************
*     File Name           :     converter.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-12 18:29]
*     Last Modified       :     [2016-02-12 23:14]
*     Description         :     covert data formats
**********************************************************************************/

#include <string>
#include <cstdlib>
#include <iostream>

#include <lsol/pario/data_iter.h>
#include <lsol/pario/data_writer.h>
#include <lsol/util/error_code.h>

#include <cmdline/cmdline.h>

using namespace lsol;
using namespace lsol::pario;
using namespace std;

int main(int argc, char** argv) {
  cmdline::parser parser;
  parser.add<string>("input", 'i', "input data path", true);
  parser.add<string>("input_type", 's', "input data type", true);
  parser.add<string>("output", 'o', "output data path", true);
  parser.add<string>("output_type", 'd', "output data type", true);

  parser.parse_check(argc, argv);

  string src_path = parser.get<string>("input");
  string src_type = parser.get<string>("input_type");
  string dst_path = parser.get<string>("output");
  string dst_type = parser.get<string>("output_type");

  DataIter iter;
  int ret = iter.AddReader(src_path, src_type);
  if (ret != Status_OK) return ret;

  DataWriter* writer = DataWriter::Create(dst_type);
  if (writer == nullptr) {
    ret = Status_Invalid_Argument;
    return ret;
  }
  ret = writer->Open(dst_path);
  if (ret != Status_OK) return ret;

  MiniBatch* mb = nullptr;

  if (dst_type == "csv") {
    // for csv, get extra info
    fprintf(stdout, "figuring out feature dimension\n");
    index_t feat_dim = 0;
    while (true) {
      mb = iter.Next(mb);
      if (mb == nullptr) break;

      for (int i = 0; i < mb->size(); ++i) {
        DataPoint& pt = (*mb)[i];
        if (feat_dim < pt.dim()) feat_dim = pt.dim();
      }
    }
    writer->SetExtraInfo((char*)(&feat_dim));
    if (feat_dim == 0) {
      fprintf(stderr, "figuring out feature dimension failed\n");
      return Status_Invalid_Format;
    }
    ret = iter.AddReader(src_path, src_type);
    if (ret != Status_OK) return ret;
  }
  int data_num = 0;
  int print_thresh = 1000;
  while (true) {
    mb = iter.Next(mb);
    if (mb == nullptr) break;
    data_num += mb->size();
    if (data_num % 1000 > print_thresh) {
      fprintf(stdout, "%d examples converted\r", data_num);
      print_thresh += 1000;
    }

    for (int i = 0; i < mb->size(); ++i) {
      writer->Write((*mb)[i]);
    }
  }
  fprintf(stdout, "%d examples converted\n", data_num);
  writer->Close();
  delete writer;
  return ret;
}
