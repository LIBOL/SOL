set(src_dirs util math pario loss model model/olm)
foreach (src_dir ${src_dirs})
    file(GLOB ${src_dir}_headers
        "${PROJECT_SOURCE_DIR}/include/lsol/${src_dir}/*.h"
        "${PROJECT_SOURCE_DIR}/include/lsol/${src_dir}/*.hpp"
        )

    file(GLOB ${src_dir}_src
        "${PROJECT_SOURCE_DIR}/src/lsol/${src_dir}/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/lsol/${src_dir}/*.cc"
        )

    source_group("Header Files\\${src_dir}" FILES ${${src_dir}_headers})
    source_group("Source Files\\${src_dir}" FILES ${${src_dir}_src})
    list(APPEND lsol_list ${${src_dir}_headers} ${${src_dir}_src})
endforeach()

file(GLOB json_files
	"${PROJECT_SOURCE_DIR}/external/json/*.h"
	"${PROJECT_SOURCE_DIR}/external/json/*.cpp"
	)
list(APPEND lsol_list ${json_files})

add_library(lsol_core SHARED ${lsol_list}
    ${PROJECT_SOURCE_DIR}/include/lsol/lsol.h
    ${PROJECT_SOURCE_DIR}/include/lsol/c_api.h
    ${PROJECT_SOURCE_DIR}/src/lsol/c_api.cc
    )
target_link_libraries(lsol_core)
list(APPEND TARGET_LIBS lsol_core)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/data
    ${CMAKE_BINARY_DIR}/data)
