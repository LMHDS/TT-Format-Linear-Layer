cmake_minimum_required(VERSION 3.0)
project(sdaccel_mm)

include(${CMAKE_SOURCE_DIR}/config.cmake)

# Vitis HLS Configuration
set(VITIS_HLS_ROOT $ENV{XILINX_HLS})
if(NOT VITIS_HLS_ROOT)
    message(FATAL_ERROR "XILINX_HLS environment variable not set")
endif()

# HLS libraries
set(VITIS_HLS_LIBS
    ${VITIS_HLS_ROOT}/lib/lnx64.o/libhlsmath.so
    ${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1/libIp_floating_point_v7_1_bitacc_cmodel.so
    ${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1/libgmp.so
    ${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1/libmpfr.so
    /opt/xilinx/xrt/lib/libxilinxopencl.so
    stdc++
    m
    pthread
)

# Set module path
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake
    ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

# Platform and Target Options
set(MM_PLATFORM "xilinx_u250_gen3x16_xdma_4_1_202210_1" CACHE STRING "Platform string for Vitis.")
set(MM_CLOCK_INTERNAL "300" CACHE INTERNAL "")
set(MM_ENABLE_PROFILING OFF CACHE BOOL "Collect profiling information.")
set(MM_ENABLE_DEBUGGING OFF CACHE BOOL "Inject debugging cores to design.")
set(MM_VITIS_FLAGS "" CACHE STRING "Extra flags for Vitis.")
set(MM_POWER_METER OFF CACHE BOOL "Enable Corsair power meter.")
set(MM_ENABLE_BLAS ON CACHE BOOL "Use BLAS library for validation if available.")

# Data type configuration
set(MM_DATA_TYPE "float" CACHE STRING "Matrix data type (float/half).")
option(USE_HALF_PRECISION "Use half precision floating point" OFF)

if(USE_HALF_PRECISION)
    set(MM_DATA_TYPE "half")
    add_definitions(-DMM_HALF_PRECISION)
endif()

# TT-Linear Configuration Options
set(TT_NUM_CORES 4 CACHE STRING "Number of TT cores")
if(TT_NUM_CORES LESS 1)
    message(FATAL_ERROR "TT_NUM_CORES must be at least 1")
endif()

set(TT_INPUT_SIZE 256 CACHE STRING "TT input dimension size")
set(TT_OUTPUT_SIZE 256 CACHE STRING "TT output dimension size")
set(TT_MAX_RANK 64 CACHE STRING "Maximum TT rank")

# TT Dimension Decomposition Configuration
option(TT_USE_CUSTOM_MODES "Use custom mode decomposition" OFF)
if(TT_USE_CUSTOM_MODES)
    set(TT_INPUT_MODES "4,4,4,4" CACHE STRING "Custom input modes (comma-separated)")
    set(TT_OUTPUT_MODES "4,4,4,4" CACHE STRING "Custom output modes (comma-separated)")
    set(TT_RANKS "1,64,64,64,1" CACHE STRING "Custom TT ranks (comma-separated)")
endif()


# Memory Configuration
set(MM_MEMORY_BUS_WIDTH_N 64 CACHE STRING "Width of memory bus in bytes in N.")
set(MM_MEMORY_BUS_WIDTH_K 64 CACHE STRING "Width of memory bus in bytes in K.")
set(MM_MEMORY_BUS_WIDTH_M 64 CACHE STRING "Width of memory bus in bytes in M.")
set(MM_DYNAMIC_SIZES ON CACHE BOOL "Use dynamic matrix dimension sizes.")
set(MM_MEMORY_TILE_SIZE_N 256 CACHE STRING "Tile size of outer memory tile in N.")
set(MM_MEMORY_TILE_SIZE_M 256 CACHE STRING "Tile size of outer memory tile in M.")
set(MM_PARALLELISM_N 32 CACHE STRING "Number of parallel compute in N.")
set(MM_PARALLELISM_M 8 CACHE STRING "Number of parallel compute in M.")
set(MM_TRANSPOSED_A OFF CACHE BOOL "Assume the input matrix A is transposed.")
set(MM_TRANSPOSE_WIDTH 64 CACHE STRING "Burst width when transposing reads from A.")
set(MM_TWO_DIMMS OFF CACHE BOOL "Use two DDR DIMMs instead of one")
set(MM_MAP_OP "Multiply" CACHE STRING "Map operation between A and B matrices.")
set(MM_REDUCE_OP "Add" CACHE STRING "Reduction operation to write back to C.")
set(MM_ADD_RESOURCE OFF CACHE STRING "")
set(MM_MULT_RESOURCE OFF CACHE STRING "")

# Function to validate TT modes
function(validate_tt_modes TOTAL_SIZE MODES VALID)
    string(REPLACE "," ";" MODE_LIST "${MODES}")
    set(product 1)
    foreach(mode IN LISTS MODE_LIST)
        math(EXPR product "${product} * ${mode}")
    endforeach()
    if(product EQUAL TOTAL_SIZE)
        set(${VALID} TRUE PARENT_SCOPE)
    else()
        set(${VALID} FALSE PARENT_SCOPE)
        message(STATUS "Dimension verification failed: product=${product}, expected=${TOTAL_SIZE}")
    endif()
endfunction()

# Function to generate TT core configurations
function(generate_tt_core_configs)
    if(TT_USE_CUSTOM_MODES)
        string(REPLACE "," ";" INPUT_MODES_LIST "${TT_INPUT_MODES}")
        string(REPLACE "," ";" OUTPUT_MODES_LIST "${TT_OUTPUT_MODES}")
        string(REPLACE "," ";" TT_RANKS_LIST "${TT_RANKS}")
    else()
        set(INPUT_MODES_LIST "8;8;8;2")
        set(OUTPUT_MODES_LIST "8;8;4;2")
        set(TT_RANKS_LIST "1;64;64;64;1")
    endif()

    # 设置TTCore维度
    list(GET INPUT_MODES_LIST 0 TT_INPUT_MODE_0)
    list(GET INPUT_MODES_LIST 1 TT_INPUT_MODE_1)
    list(GET INPUT_MODES_LIST 2 TT_INPUT_MODE_2)
    list(GET INPUT_MODES_LIST 3 TT_INPUT_MODE_3)
    
    list(GET OUTPUT_MODES_LIST 0 TT_OUTPUT_MODE_0)
    list(GET OUTPUT_MODES_LIST 1 TT_OUTPUT_MODE_1)
    list(GET OUTPUT_MODES_LIST 2 TT_OUTPUT_MODE_2)
    list(GET OUTPUT_MODES_LIST 3 TT_OUTPUT_MODE_3)
    
    list(GET TT_RANKS_LIST 0 TT_RANK_0)
    list(GET TT_RANKS_LIST 1 TT_RANK_1)
    list(GET TT_RANKS_LIST 2 TT_RANK_2)
    list(GET TT_RANKS_LIST 3 TT_RANK_3)
    list(GET TT_RANKS_LIST 4 TT_RANK_4)

    # 计算GEMM维度
    # First GEMM: [I2*I3*I4, I1] × [I1, r1*O1]
    math(EXPR TT_GEMM_N0 "${TT_INPUT_MODE_1} * ${TT_INPUT_MODE_2} * ${TT_INPUT_MODE_3}")
    set(TT_GEMM_K0 ${TT_INPUT_MODE_0})
    math(EXPR TT_GEMM_M0 "${TT_RANK_1} * ${TT_OUTPUT_MODE_0}")

    # Second GEMM: [O1*I3*I4, r1*I2] × [r1*I2, r2*O2]
    math(EXPR TT_GEMM_N1 "${TT_OUTPUT_MODE_0} * ${TT_INPUT_MODE_2} * ${TT_INPUT_MODE_3}")
    math(EXPR TT_GEMM_K1 "${TT_RANK_1} * ${TT_INPUT_MODE_1}")
    math(EXPR TT_GEMM_M1 "${TT_RANK_2} * ${TT_OUTPUT_MODE_1}")

    # Third GEMM: [O1*O2*I4, r2*I3] × [r2*I3, r3*O3]
    math(EXPR TT_GEMM_N2 "${TT_OUTPUT_MODE_0} * ${TT_OUTPUT_MODE_1} * ${TT_INPUT_MODE_3}")
    math(EXPR TT_GEMM_K2 "${TT_RANK_2} * ${TT_INPUT_MODE_2}")
    math(EXPR TT_GEMM_M2 "${TT_RANK_3} * ${TT_OUTPUT_MODE_2}")

    # Fourth GEMM: [O1*O2*O3, r3*I4] × [r3*I4, O4]
    math(EXPR TT_GEMM_N3 "${TT_OUTPUT_MODE_0} * ${TT_OUTPUT_MODE_1} * ${TT_OUTPUT_MODE_2}")
    math(EXPR TT_GEMM_K3 "${TT_RANK_3} * ${TT_INPUT_MODE_3}")
    math(EXPR TT_GEMM_M3 "${TT_RANK_4} * ${TT_OUTPUT_MODE_3}")

    # 设置TTCore维度变量到父作用域
    set(TT_INPUT_MODE_0 ${TT_INPUT_MODE_0} PARENT_SCOPE)
    set(TT_INPUT_MODE_1 ${TT_INPUT_MODE_1} PARENT_SCOPE)
    set(TT_INPUT_MODE_2 ${TT_INPUT_MODE_2} PARENT_SCOPE)
    set(TT_INPUT_MODE_3 ${TT_INPUT_MODE_3} PARENT_SCOPE)

    set(TT_OUTPUT_MODE_0 ${TT_OUTPUT_MODE_0} PARENT_SCOPE)
    set(TT_OUTPUT_MODE_1 ${TT_OUTPUT_MODE_1} PARENT_SCOPE)
    set(TT_OUTPUT_MODE_2 ${TT_OUTPUT_MODE_2} PARENT_SCOPE)
    set(TT_OUTPUT_MODE_3 ${TT_OUTPUT_MODE_3} PARENT_SCOPE)

    set(TT_RANK_0 ${TT_RANK_0} PARENT_SCOPE)
    set(TT_RANK_1 ${TT_RANK_1} PARENT_SCOPE)
    set(TT_RANK_2 ${TT_RANK_2} PARENT_SCOPE)
    set(TT_RANK_3 ${TT_RANK_3} PARENT_SCOPE)
    set(TT_RANK_4 ${TT_RANK_4} PARENT_SCOPE)

    # 设置GEMM维度变量到父作用域
    set(TT_GEMM_N0 ${TT_GEMM_N0} PARENT_SCOPE)
    set(TT_GEMM_K0 ${TT_GEMM_K0} PARENT_SCOPE)
    set(TT_GEMM_M0 ${TT_GEMM_M0} PARENT_SCOPE)

    set(TT_GEMM_N1 ${TT_GEMM_N1} PARENT_SCOPE)
    set(TT_GEMM_K1 ${TT_GEMM_K1} PARENT_SCOPE)
    set(TT_GEMM_M1 ${TT_GEMM_M1} PARENT_SCOPE)

    set(TT_GEMM_N2 ${TT_GEMM_N2} PARENT_SCOPE)
    set(TT_GEMM_K2 ${TT_GEMM_K2} PARENT_SCOPE)
    set(TT_GEMM_M2 ${TT_GEMM_M2} PARENT_SCOPE)

    set(TT_GEMM_N3 ${TT_GEMM_N3} PARENT_SCOPE)
    set(TT_GEMM_K3 ${TT_GEMM_K3} PARENT_SCOPE)
    set(TT_GEMM_M3 ${TT_GEMM_M3} PARENT_SCOPE)

    # 添加维度验证
    math(EXPR total_input "${TT_INPUT_MODE_0} * ${TT_INPUT_MODE_1} * ${TT_INPUT_MODE_2} * ${TT_INPUT_MODE_3}")
    math(EXPR total_output "${TT_OUTPUT_MODE_0} * ${TT_OUTPUT_MODE_1} * ${TT_OUTPUT_MODE_2} * ${TT_OUTPUT_MODE_3}")
    
    if(NOT ${total_input} EQUAL ${TT_INPUT_SIZE})
        message(FATAL_ERROR "Input modes product (${total_input}) does not match input size (${TT_INPUT_SIZE})")
    endif()
    
    if(NOT ${total_output} EQUAL ${TT_OUTPUT_SIZE})
        message(FATAL_ERROR "Output modes product (${total_output}) does not match output size (${TT_OUTPUT_SIZE})")
    endif()

# Generate memory width validation and conversion logic
# TT Core Memory Configurations
foreach(core_idx RANGE 0 3)
    set(TT_CORE${core_idx}_MEMORY_WIDTH_N 64 CACHE STRING "Memory width N for core ${core_idx}")
    set(TT_CORE${core_idx}_MEMORY_WIDTH_K 64 CACHE STRING "Memory width K for core ${core_idx}")
    set(TT_CORE${core_idx}_MEMORY_WIDTH_M 64 CACHE STRING "Memory width M for core ${core_idx}")
    
    set(TT_CORE${core_idx}_COMPUTE_TILE_N 1 CACHE STRING "Compute tile size N for core ${core_idx}")
    set(TT_CORE${core_idx}_COMPUTE_TILE_M 8 CACHE STRING "Compute tile size M for core ${core_idx}")
    
    set(TT_CORE${core_idx}_MEMORY_ALIGNMENT 4096 CACHE STRING "Memory alignment for core ${core_idx}")
endforeach()

# 验证每个核心的配置
function(validate_core_config core_idx)
    math(EXPR width_check_m 
         "${TT_CORE${core_idx}_MEMORY_WIDTH_M} % ${TT_CORE${core_idx}_COMPUTE_TILE_M}")
    if(NOT width_check_m EQUAL 0)
        message(WARNING "Core ${core_idx}: Memory width M must be divisible by compute tile size M")
    endif()
    
    math(EXPR width_check_n
         "${TT_CORE${core_idx}_MEMORY_WIDTH_N} % ${TT_CORE${core_idx}_COMPUTE_TILE_N}")
    if(NOT width_check_n EQUAL 0)
        message(WARNING "Core ${core_idx}: Memory width N must be divisible by compute tile size N")
    endif()
endfunction()

foreach(core_idx RANGE 0 3)
    validate_core_config(${core_idx})
endforeach()


endfunction()

# Generate TT core configurations
generate_tt_core_configs()

# Configure header files
configure_file(include/Config.h.in ${CMAKE_BINARY_DIR}/Config.h )

# Find required packages
find_package(Vitis REQUIRED)
find_package(Threads REQUIRED)

if(MM_ENABLE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        add_definitions(-DMM_HAS_BLAS)
    endif()
endif()

# Include directories
include_directories(
    include 
    ${CMAKE_BINARY_DIR} 
    SYSTEM 
    hlslib/include 
    ${Vitis_INCLUDE_DIRS}
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

if(MM_DYNAMIC_SIZES)
    add_definitions(-DMM_DYNAMIC_SIZES)
endif()

if(MM_TRANSPOSED_A)
    add_definitions(-DMM_TRANSPOSED_A)
endif()

# GEMM Kernel Library
set(MM_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/Compute.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Top.cpp)

add_library(mmkernel ${MM_KERNEL_SRC})

target_link_libraries(mmkernel 
    PRIVATE 
    ${CMAKE_THREAD_LIBS_INIT}
)

# Set runtime link path for mmkernel
set_target_properties(mmkernel PROPERTIES
    INSTALL_RPATH "${VITIS_HLS_ROOT}/lib/lnx64.o;${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1"
    BUILD_WITH_INSTALL_RPATH TRUE
    LINK_FLAGS "-Wl,--no-as-needed"
)

# Half precision support
if(USE_HALF_PRECISION)
    target_link_libraries(mmkernel 
        PRIVATE 
        ${VITIS_HLS_LIBS}
    )
endif()

# TT-Linear Library
add_library(tt_linear
    src/tt_linear/tt_core.cpp
    src/tt_linear/tt_linear.cpp
)

target_link_libraries(tt_linear
    PUBLIC
        mmkernel
)

target_compile_definitions(tt_linear
    PUBLIC
    TT_NUM_CORES=${TT_NUM_CORES}
)

if(USE_HALF_PRECISION)
    target_link_libraries(tt_linear
        PRIVATE
        ${VITIS_HLS_LIBS}
    )
endif()

# TT-Linear Test Program
add_executable(tt_linear_test
    test/tt_linear_test.cpp
)

target_link_libraries(tt_linear_test
    PRIVATE
        tt_linear
)

# Enable testing
enable_testing()
add_test(NAME tt_test COMMAND tt_linear_test)
