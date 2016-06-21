set(src_dirs util pario model)
foreach (src_dir ${src_dirs})
    file(GLOB ${src_dir}_src
        "${PROJECT_SOURCE_DIR}/test/${src_dir}/*.cpp"
        "${PROJECT_SOURCE_DIR}/test/${src_dir}/*.cc"
        )

    foreach(test_src ${${src_dir}_src})
        get_filename_component(tgt_name ${test_src} NAME_WE)
        add_executable(${tgt_name} ${test_src})
        target_link_libraries(${tgt_name} lsol)
        SET_PROPERTY(TARGET ${tgt_name} PROPERTY FOLDER "test/${src_dir}")
        list(APPEND test_targets ${tgt_name})
    endforeach()
endforeach()
