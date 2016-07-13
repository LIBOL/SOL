cmake_minimum_required(VERSION 2.8.12)

project(LIBSOL)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(ARCHIVE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if (NOT PREFIX)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/dist)
else()
    set(CMAKE_INSTALL_PREFIX ${PREFIX})
endif()

include(compiler)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external
    )

#include(util)
#include(pario)
include(lsol)

include(tools)
include(test)

include(summary)

install(TARGETS ${TARGET_LIBS}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib
    )

install(DIRECTORY include
    DESTINATION .
    )
