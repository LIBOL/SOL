file(GLOB util_headers
	"${PROJECT_SOURCE_DIR}/include/lsol/util/*.h"
	"${PROJECT_SOURCE_DIR}/include/lsol/util/*.hpp"
	)

file(GLOB util_src
	"${PROJECT_SOURCE_DIR}/src/lsol/util/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/lsol/util/*.cc"
	)

source_group("Header Files" FILES ${util_headers})
source_group("Source Files" FILES ${util_src})

file(GLOB math_headers
	"${PROJECT_SOURCE_DIR}/include/lsol/math/*.h"
	"${PROJECT_SOURCE_DIR}/include/lsol/math/*.hpp"
	)
source_group("Header Files\\math" FILES ${math_headers})

add_library(lsol_util SHARED ${util_headers} ${util_src} ${math_headers})
list(APPEND TARGET_LIBS lsol_util)
