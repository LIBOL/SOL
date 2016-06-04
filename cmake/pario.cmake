file(GLOB pario_headers
	"${PROJECT_SOURCE_DIR}/include/lsol/pario/*.h"
	"${PROJECT_SOURCE_DIR}/include/lsol/pario/*.hpp"
	)

file(GLOB pario_src
	"${PROJECT_SOURCE_DIR}/src/lsol/pario/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/lsol/pario/*.cc"
	)

source_group("Header Files\\pario" FILES ${pario_headers})
source_group("Source Files\\pario" FILES ${pario_src})

set(src_list ${src_list} ${pario_headers} ${pario_src})

add_library(lsol SHARED ${src_list})
set_target_properties(lsol PROPERTIES COMPILE_DEFINITIONS  "LSOL_EXPORTS")
list(APPEND TARGETS lsol
    )

