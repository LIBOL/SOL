file(GLOB tools_src_list
	"${PROJECT_SOURCE_DIR}/tools/*.cpp"
	"${PROJECT_SOURCE_DIR}/tools/*.cc"
	)

foreach(tool_src ${tools_src_list})
	get_filename_component(tgt_name ${tool_src} NAME_WE)
  add_executable(${tgt_name}-bin ${tool_src})
    target_link_libraries(${tgt_name}-bin ${LIBS} lsol)
	SET_PROPERTY(TARGET ${tgt_name}-bin PROPERTY FOLDER "tools")
	list(APPEND tools_targets ${tgt_name}-bin)
endforeach()

install(TARGETS ${tools_targets}
	RUNTIME DESTINATION tools
	LIBRARY DESTINATION tools
	ARCHIVE DESTINATION tools
	)

install(TARGETS ${TARGETS}
    RUNTIME DESTINATION tools
    LIBRARY DESTINATION tools
    ARCHIVE DESTINATION tools
    )

