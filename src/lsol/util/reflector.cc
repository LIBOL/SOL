/*********************************************************************************
*     File Name           :     reflector.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-23 15:09]
*     Last Modified       :     [2016-05-15 01:42]
*     Description         :
**********************************************************************************/

#include "lsol/util/reflector.h"

#include <map>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

using namespace std;

namespace lsol {

ClassInfo::ClassInfo(const std::string& name, void* func,
                     const std::string& descr)
    : name_(name), create_func_(func), descr_(descr) {
  ClassFactory::Register(this);
}

void ClassFactory::Register(ClassInfo* class_info) {
  ClsInfoMapType& cls_info_map = ClassInfoMap();
  if (cls_info_map.find(class_info->name()) == cls_info_map.end()) {
    cls_info_map[class_info->name()] = class_info;
  } else {
    fprintf(stderr, "%s already exists!\n", class_info->name().c_str());
    exit(1);
  }
}

ClassFactory::ClsInfoMapType& ClassFactory::ClassInfoMap() {
  static ClsInfoMapType class_info_map;
  return class_info_map;
}

}  // namespace shentu
