file(GLOB test_src_list
	"${PROJECT_SOURCE_DIR}/test/*.cpp"
	"${PROJECT_SOURCE_DIR}/test/*.cc"
	)

foreach(test_src ${test_src_list})
	get_filename_component(tgt_name ${test_src} NAME_WE)
	add_executable(${tgt_name} ${test_src})
    target_link_libraries(${tgt_name} ${LIBS} lsol)
	SET_PROPERTY(TARGET ${tgt_name} PROPERTY FOLDER "test")
	list(APPEND test_targets ${tgt_name})
endforeach()

install(TARGETS ${test_targets}
	RUNTIME DESTINATION test
	LIBRARY DESTINATION test
	ARCHIVE DESTINATION test
	)

install(TARGETS ${TARGETS}
    RUNTIME DESTINATION test
    LIBRARY DESTINATION test
    ARCHIVE DESTINATION test
    )

