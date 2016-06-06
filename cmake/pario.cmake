file(GLOB pario_headers
	"${PROJECT_SOURCE_DIR}/include/lsol/pario/*.h"
	"${PROJECT_SOURCE_DIR}/include/lsol/pario/*.hpp"
	)

file(GLOB pario_src
	"${PROJECT_SOURCE_DIR}/src/lsol/pario/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/lsol/pario/*.cc"
	)

source_group("Header Files" FILES ${pario_headers})
source_group("Source Files" FILES ${pario_src})


add_library(lsol_pario SHARED ${pario_headers} ${pario_src})
target_link_libraries(lsol_pario lsol_util)
list(APPEND TARGET_LIBS lsol_pario)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/data
    ${CMAKE_BINARY_DIR}/data)
