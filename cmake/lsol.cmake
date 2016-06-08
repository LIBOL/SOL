set(src_dirs loss model model/olm optimizer)
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


add_library(lsol_core SHARED ${lsol_list} ${PROJECT_SOURCE_DIR}/include/lsol/lsol.h)
target_link_libraries(lsol_core lsol_pario lsol_util)
list(APPEND TARGET_LIBS lsol_core)
