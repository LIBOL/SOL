include(ExternalProject)

ExternalProject_Add(cmdline
    PREFIX ${PROJECT_SOURCE_DIR}/external
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/external
    TMP_DIR ${CMAKE_BINARY_DIR}/tmp
    GIT_REPOSITORY git@github.com:yuewu001/cmdline.git
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
