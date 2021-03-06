# operators
set(DIR_OPERATORS_SRCS "")
macro(mercury_add_layer class)
    string(TOLOWER ${class} name)

    # WITH_LAYER_xxx option
    if(${ARGC} EQUAL 2)
        option(WITH_LAYER_${name} "build with layer ${name}" ${ARGV1})
    else()
        option(WITH_LAYER_${name} "build with layer ${name}" ON)
    endif()

    message("WITH_LAYER_${name} = ${WITH_LAYER_${name}}")

    if(WITH_LAYER_${name})
        list(APPEND DIR_OPERATORS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp")

        # look for arch specific implementation and append source
        # optimized implementation for armv7 aarch64
        if((ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
            OR (ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
            OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "armv7"))
            OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "arm64"))
            OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "armv7;arm64")))
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/arm/${name}_arm.cpp")
                list(APPEND DIR_OPERATORS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/arm/${name}_arm.cpp")
                set(WITH_LAYER_${name}_arm 1)
            endif()
        elseif(USE_CUDA)
        	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/${name}_cudnn.cpp")
                	list(APPEND DIR_OPERATORS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/${name}_cudnn.cpp")
                	set(WITH_LAYER_${name}_cuda 1)
            	elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/${name}_cuda.cu")
            		list(APPEND DIR_OPERATORS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/${name}_cuda.cu")
                	set(WITH_LAYER_${name}_cuda 1)
           	 endif()
        else()
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/x86/${name}_x86.cpp")
                list(APPEND DIR_OPERATORS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/x86/${name}_x86.cpp")
                set(WITH_LAYER_${name}_x86 1)
            endif()
        endif()
    endif()

    # generate layer_declaration and layer_registry file
    #CMAKE_CURRENT_BINARY_DIR
    if(WITH_LAYER_${name})
        if(WITH_LAYER_${name}_arm)
            file(APPEND ${PROJECT_SOURCE_DIR}/layer_declaration.h
                "extern Layer* ${class}_arm_layer_creator();\n")
            file(APPEND ${PROJECT_SOURCE_DIR}/layer_registry.h
                "#if USE_STRING\n{\"${class}\",${class}_arm_layer_creator},\n#else\n{${class}_arm_layer_creator},\n#endif\n")
        elseif(WITH_LAYER_${name}_cuda)
            file(APPEND ${PROJECT_SOURCE_DIR}/layer_declaration.h
                "extern Layer* ${class}_cuda_layer_creator();\n")
            file(APPEND ${PROJECT_SOURCE_DIR}/layer_registry.h
                "#if USE_STRING\n{\"${class}\",${class}_cuda_layer_creator},\n#else\n{${class}_cuda_layer_creator},\n#endif\n")
        elseif(WITH_LAYER_${name}_x86)
            file(APPEND ${PROJECT_SOURCE_DIR}/layer_declaration.h
                "extern Layer* ${class}_x86_layer_creator();\n")
            file(APPEND ${PROJECT_SOURCE_DIR}/layer_registry.h
                "#if USE_STRING\n{\"${class}\",${class}_x86_layer_creator},\n#else\n{${class}_x86_layer_creator},\n#endif\n")
        else()
            file(APPEND ${PROJECT_SOURCE_DIR}/layer_declaration.h
                "extern Layer* ${class}_layer_creator();\n")
            file(APPEND ${PROJECT_SOURCE_DIR}/layer_registry.h
                "#if USE_STRING\n{\"${class}\",${class}_layer_creator},\n#else\n{${class}_layer_creator},\n#endif\n")
        endif()
    else()
        file(APPEND ${PROJECT_SOURCE_DIR}/layer_registry.h "#if USE_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()

    # generate layer_type_enum file
    file(APPEND ${PROJECT_SOURCE_DIR}/layer_type_enum.h "${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()

# create new
file(REMOVE ${PROJECT_SOURCE_DIR}/layer_declaration.h)
file(REMOVE ${PROJECT_SOURCE_DIR}/layer_registry.h)
file(REMOVE ${PROJECT_SOURCE_DIR}/layer_type_enum.h)
set(__LAYER_TYPE_ENUM_INDEX 0)

# layer implementation
mercury_add_layer(AbsVal)
mercury_add_layer(ArgMax OFF)
mercury_add_layer(BatchNorm)
mercury_add_layer(Bias)
mercury_add_layer(BNLL)
mercury_add_layer(Concat)
mercury_add_layer(Convolution)
mercury_add_layer(Crop)
mercury_add_layer(Deconvolution)
mercury_add_layer(Dropout)
mercury_add_layer(Eltwise)
mercury_add_layer(ELU)
mercury_add_layer(Embed OFF)
mercury_add_layer(Exp)
mercury_add_layer(Flatten)
mercury_add_layer(InnerProduct)
mercury_add_layer(Input)
mercury_add_layer(Log)
mercury_add_layer(LRN)
mercury_add_layer(MemoryData)
mercury_add_layer(MVN)
mercury_add_layer(Pooling)
mercury_add_layer(Power)
mercury_add_layer(PReLU)
mercury_add_layer(Proposal)
mercury_add_layer(Reduction)
mercury_add_layer(ReLU)
mercury_add_layer(Reshape)
mercury_add_layer(ROIPooling)
mercury_add_layer(Scale)
mercury_add_layer(Sigmoid)
mercury_add_layer(Slice)
mercury_add_layer(Softmax)
mercury_add_layer(Split)
mercury_add_layer(SPP OFF)
mercury_add_layer(TanH)
mercury_add_layer(Threshold)
mercury_add_layer(Tile OFF)
mercury_add_layer(RNN OFF)
mercury_add_layer(LSTM OFF)
mercury_add_layer(BinaryOp)
mercury_add_layer(UnaryOp)
mercury_add_layer(ConvolutionDepthWise)
mercury_add_layer(Padding)
mercury_add_layer(Squeeze)
mercury_add_layer(ExpandDims)
mercury_add_layer(Normalize)
mercury_add_layer(Permute)
mercury_add_layer(PriorBox)
mercury_add_layer(DetectionOutput)
mercury_add_layer(Interp)
mercury_add_layer(DeconvolutionDepthWise)
mercury_add_layer(ShuffleChannel)

#aux_source_directory(. DIR_OPERATORS_SRCS)
message(STATUS "operator source file: ${DIR_OPERATORS_SRCS}")
add_library(operators ${DIR_OPERATORS_SRCS})
