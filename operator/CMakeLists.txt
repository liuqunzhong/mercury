# operators
aux_source_directory(. DIR_OPERATORS_SRCS)
message(STATUS "operator source file: ${DIR_OPERATORS_SRCS}")
add_library(operators ${DIR_OPERATORS_SRCS})
