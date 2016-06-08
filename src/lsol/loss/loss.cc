/*********************************************************************************
*     File Name           :     loss.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-14 23:23]
*     Last Modified       :     [2016-02-14 23:24]
*     Description         :     base class for loss functions
**********************************************************************************/

#include "lsol/loss/loss.h"

namespace lsol {
namespace loss {

Loss* Loss::Create(const std::string& type) {
  auto create_func = CreateObject<Loss>(std::string(type) + "_loss");
  return create_func == nullptr ? nullptr : create_func();
}

}  // namespace loss
}  // namespace lsol
