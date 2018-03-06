
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)


include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/layer)

add_executable(Test_deconv_eigen Test_deconv_eigen.cpp)
add_executable(Test_convolution conv_benchmark.cpp)

set_property(TARGET Test_convolution PROPERTY COMPILE_FLAGS "-fpie")
set_property(TARGET Test_convolution PROPERTY LINK_FLAGS "-pie")

target_link_libraries(Test_deconv_eigen ncnn)
target_link_libraries(Test_convolution ncnn)
