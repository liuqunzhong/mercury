# framework
aux_source_directory(. DIR_FRAMEWOEK_SRCS)
message(STATUS "framework source files: ${DIR_FRAMEWORK_SRCS}")
add_library(framework ${DIR_FRAMEWOEK_SRCS})
