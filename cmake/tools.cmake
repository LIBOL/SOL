set(TOOLS_DIR ${PROJECT_SOURCE_DIR}/tools)

add_executable(converter ${TOOLS_DIR}/converter.cc)
target_link_libraries(converter lsol_pario lsol_util)
list(APPEND tools_targets converter)

foreach(tgt_name ${tools_targets})
SET_PROPERTY(TARGET ${tgt_name} PROPERTY FOLDER "tools")
endforeach()

install(TARGETS ${tools_targets}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION bin
	ARCHIVE DESTINATION lib
	)
