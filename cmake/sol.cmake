set(src_dirs util math pario loss model model/olm)
foreach (src_dir ${src_dirs})
    file(GLOB ${src_dir}_headers
        "${PROJECT_SOURCE_DIR}/include/sol/${src_dir}/*.h"
        "${PROJECT_SOURCE_DIR}/include/sol/${src_dir}/*.hpp"
        )

    file(GLOB ${src_dir}_src
        "${PROJECT_SOURCE_DIR}/src/sol/${src_dir}/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/sol/${src_dir}/*.cc"
        )

    STRING(REGEX REPLACE "/" "\\\\" win_src_dir ${src_dir})
    source_group("Header Files\\${win_src_dir}" FILES ${${src_dir}_headers})
    source_group("Source Files\\${win_src_dir}" FILES ${${src_dir}_src})
    list(APPEND sol_list ${${src_dir}_headers} ${${src_dir}_src})
endforeach()

file(GLOB json_files
	"${PROJECT_SOURCE_DIR}/external/json/*.h"
	"${PROJECT_SOURCE_DIR}/external/json/*.cpp"
	)
list(APPEND sol_list ${json_files})

add_library(sol SHARED ${sol_list}
    ${PROJECT_SOURCE_DIR}/include/sol/sol.h
    ${PROJECT_SOURCE_DIR}/include/sol/c_api.h
    ${PROJECT_SOURCE_DIR}/include/sol/tools.h
    ${PROJECT_SOURCE_DIR}/src/sol/c_api.cc
    ${PROJECT_SOURCE_DIR}/src/sol/tools.cc
    )
target_link_libraries(sol ${LINK_LIBS})
list(APPEND TARGET_LIBS sol)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/data
    ${CMAKE_BINARY_DIR}/data)
