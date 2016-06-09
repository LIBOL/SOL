set(TOOLS_DIR ${PROJECT_SOURCE_DIR}/tools)

add_executable(converter ${TOOLS_DIR}/converter.cc)
target_link_libraries(converter lsol_core)
list(APPEND tools_targets converter)

add_executable(lsol ${TOOLS_DIR}/lsol.cc)
target_link_libraries(lsol lsol_core)
list(APPEND tools_targets lsol)

add_executable(lsol_c ${TOOLS_DIR}/lsol_c.cc)
target_link_libraries(lsol_c lsol_core)
list(APPEND tools_targets lsol_c)


foreach(tgt_name ${tools_targets})
SET_PROPERTY(TARGET ${tgt_name} PROPERTY FOLDER "tools")
endforeach()

install(TARGETS ${tools_targets}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION bin
	ARCHIVE DESTINATION lib
	)
