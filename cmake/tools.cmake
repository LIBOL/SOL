set(TOOLS_DIR ${PROJECT_SOURCE_DIR}/tools)

file(GLOB tool_list
	"${TOOLS_DIR}/*.cpp"
	"${TOOLS_DIR}/*.cc"
	)

foreach(tool_src ${tool_list})
	get_filename_component(tgt_name ${tool_src} NAME_WE)
	add_executable(${tgt_name} ${tool_src})
    target_link_libraries(${tgt_name} lsol ${LINK_LIBS})
    SET_PROPERTY(TARGET ${tgt_name} PROPERTY FOLDER "tools")
	list(APPEND tools_targets ${tgt_name})
endforeach()

install(TARGETS ${tools_targets}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION bin
	ARCHIVE DESTINATION lib
	)
