/*********************************************************************************
*     File Name           :     global_config.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-29 11:23]
*     Last Modified       :     [2015-12-03 14:49]
*     Description         :     Global Configurations
**********************************************************************************/

#ifndef LSOL_UTIL_GLOBAL_CONFIG_H__
#define LSOL_UTIL_GLOBAL_CONFIG_H__

#include <cstdint>
#include <type_traits>

namespace lsol {

#if (defined WIN32 || defined _WIN32 || defined WINCE)
#ifdef LSOL_EXPORTS
#undef LSOL_EXPORTS
#define LSOL_EXPORTS __declspec(dllexport)
#else
#define LSOL_EXPORTS __declspec(dllimport)
#endif
#else
#undef LSOL_EXPORTS
#define LSOL_EXPORTS
#endif

#ifndef FeatType
#define FeatType float
#endif

#ifndef IndexType
#define IndexType uint32_t
#endif

#ifndef LabelType
#define LabelType int32_t
#endif

/// \brief  only float or double type are allowed for features
template <typename T>
struct feat_type_traits {
  typedef typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value,
                                  T>::type type;
};

typedef feat_type_traits<FeatType>::type real_t;

/// \brief  only uint16_t, uint32_t or uint64_t type are allowed for features
template <typename T>
struct index_type_traits {
  typedef typename std::enable_if<std::is_same<T, uint16_t>::value ||
                                      std::is_same<T, uint32_t>::value ||
                                      std::is_same<T, uint64_t>::value,
                                  T>::type type;
};

typedef index_type_traits<IndexType>::type index_t;

/// \brief  only char, short, int32, int64 are allowed for features
template <typename T>
struct label_type_traits {
  typedef typename std::enable_if<
      std::is_same<T, char>::value || std::is_same<T, short>::value ||
          std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value,
      T>::type type;
};

typedef label_type_traits<LabelType>::type label_t;

}  // namespace lsol

#endif
