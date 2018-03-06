
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(Test_deconv_eigen Test_deconv_eigen.cpp)

target_link_libraries(Test_deconv_eigen ncnn)
