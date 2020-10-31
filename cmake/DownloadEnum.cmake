CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)

PROJECT(enum-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(enum
	URL https://pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz
	URL_HASH SHA256=8ad8c4783bf61ded74527bffb48ed9b54166685e4230386a9ed9b1279e2df5b1
	SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/enum"
	BINARY_DIR "${CONFU_DEPENDENCIES_BINARY_DIR}/enum"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)
