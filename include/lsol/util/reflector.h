/*************************************************************************
  > File Name: reflector.h
  > Copyright (C) 2014 Yue Wu<yuewu@outlook.com>
  > Created Time: 2014/5/12 Monday 16:21:00
  > Functions: C++ reflector
 ************************************************************************/
#ifndef CXX_SELF_CUSTOMIZED_RELFECTOR_H__
#define CXX_SELF_CUSTOMIZED_RELFECTOR_H__

#include <string>
#include <map>

#include <lsol/util/types.h>

namespace lsol {

/// \brief  Information of a class, its name, creator, and description
class LSOL_EXPORTS ClassInfo {
 public:
  ClassInfo(const std::string& name, void* func, const std::string& descr = "");

 public:
  const std::string& name() const { return this->name_; }
  void* create_func() const { return this->create_func_; }
  const std::string& descr() const { return this->descr_; }

 private:
  std::string name_;
  void* create_func_;
  std::string descr_;
};

/// \brief  Factory of classes
class LSOL_EXPORTS ClassFactory {
 public:
  typedef std::map<std::string, ClassInfo*> ClsInfoMapType;

  /// \brief  Called outside of Registry to register a new class to
  /// dictionary
  ///
  /// \param class_info Information about the class, including:
  //                          1. Name of the class
  //                          2. CreateFunction poniter
  //                          3. Description of the class(optional)
  static void Register(ClassInfo* class_info);

  static ClsInfoMapType& ClassInfoMap();
};

/// \brief  Create a new class according to the name of the class
///
/// \tparam ClsType Type of the class
/// \param name Name of the class
/// \param params Parameter required to create the class
///
/// \return Pointer to the created class instance
template <typename ClsType,
          typename ReturnType = typename ClsType::CreateFunction>
ReturnType CreateObject(const std::string& cls_name) {
  auto cls_info_map = ClassFactory::ClassInfoMap();
  auto iter = cls_info_map.find(cls_name);
  if (iter != cls_info_map.end()) {
    return (ReturnType((iter->second)->create_func()));
  }
  return ReturnType(nullptr);
}

#define UniqueClassName(name, suffix) name##suffix

#define DeclareReflectorBase(type, ...)                            \
 public:                                                           \
  typedef type* (*CreateFunction)(__VA_ARGS__);                    \
                                                                   \
  static type* Create(const std::string& cls_name, ##__VA_ARGS__); \
                                                                   \
 private:

#define RegisterClassReflector(type, name, descr)                              \
  type* type##_##CreateNewInstance() { return new type(); }                    \
  ClassInfo __kClassInfo_##type##__(name, (void*)(type##_##CreateNewInstance), \
                                    descr);
}  // namespace lsol

#endif  // CXX_SELF_CUSTOMIZED_RELFECTOR_H__
