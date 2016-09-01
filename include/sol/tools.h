/*********************************************************************************
*     File Name           :     tools.h
*     Created By          :     yuewu
*     Description         :     tools for  sol
**********************************************************************************/
#ifndef SOL_TOOLS_H__
#define SOL_TOOLS_H__

#include <string>
#include "sol/util/types.h"

namespace sol {
SOL_EXPORTS int analyze(const std::string& src_path,
                        const std::string& src_type,
                        const std::string& output_path);

SOL_EXPORTS int convert(const std::string& src_path,
                        const std::string& src_type,
                        const std::string& dst_path,
                        const std::string& dst_type);

SOL_EXPORTS int shuffle(const std::string& src_path,
                        const std::string& src_type,
                        const std::string& output_path,
                        const std::string& output_type);

SOL_EXPORTS int split(const std::string& src_path, const std::string& src_type,
                      int fold_num, const std::string& output_prefix,
                      const std::string& dst_type, bool shuffle);
}
#endif
