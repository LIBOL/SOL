/*********************************************************************************
*     File Name           :     tools.h
*     Created By          :     yuewu
*     Description         :     tools for  lsol
**********************************************************************************/
#ifndef LSOL_TOOLS_H__
#define LSOL_TOOLS_H__

#include <string>
#include "lsol/util/types.h"

namespace lsol {
LSOL_EXPORTS int analyze(const std::string& src_path, const std::string& src_type,
            const std::string& output_path);

LSOL_EXPORTS int convert(const std::string& src_path, const std::string& src_type,
            const std::string& dst_path, const std::string& dst_type);

LSOL_EXPORTS int shuffle(const std::string& src_path, const std::string& src_type,
            const std::string& output_path, const std::string& output_type);

LSOL_EXPORTS int split(const std::string& src_path, const std::string& src_type,
          int fold_num, const std::string& output_prefix,
          const std::string& dst_type, bool shuffle);
}
#endif
