/*********************************************************************************
*     File Name           :     ../util/error_code.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-16 23:44]
*     Last Modified       :     [2015-12-03 14:49]
*     Description         :     error code of lsos
**********************************************************************************/
#ifndef LSOL_UTIL_ERROR_CODE_H__
#define LSOL_UTIL_ERROR_CODE_H__

namespace lsol {

static const int Status_OK = 0;
static const int Status_Error = 1;
static const int Status_IO_Error = 2;
static const int Status_EndOfFile = 3;
static const int Status_Invalid_Argument = 4;
static const int Status_Invalid_Format = 5;

}  // namespace lsol

#endif  // LSOL_UTIL_ERROR_CODE_H__
