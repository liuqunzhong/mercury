### base type for mercury
aux_source_directory(. DIR_BASE_SRCS)
message(STATUS "base source files: ${DIR_BASE_SRCS}")
add_library(base ${DIR_BASE_SRCS})
