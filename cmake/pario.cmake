file(GLOB pario_headers
	"${PROJECT_SOURCE_DIR}/include/sol/pario/*.h"
	"${PROJECT_SOURCE_DIR}/include/sol/pario/*.hpp"
	)

file(GLOB pario_src
	"${PROJECT_SOURCE_DIR}/src/sol/pario/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/sol/pario/*.cc"
	)

source_group("Header Files" FILES ${pario_headers})
source_group("Source Files" FILES ${pario_src})


add_library(sol_pario SHARED ${pario_headers} ${pario_src})
target_link_libraries(sol_pario sol_util)
list(APPEND TARGET_LIBS sol_pario)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/data
    ${CMAKE_BINARY_DIR}/data)
