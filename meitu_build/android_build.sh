#!/bin/sh
set -e

current_dir=`pwd`/`dirname $0`
echo $current_dir
pushd $current_dir

export ALL_ARCHS="armeabi-v7a arm64-v8a"

#export BUILD_TYPE=MinSizeRel
export BUILD_TYPE=Release
# 静态库OFF, 动态库ON
export BUILD_SHARED_LIBS="OFF"
# 隐藏符号ON, 不隐藏符号OFF
export BUILD_HIDDEN_SYMBOL="ON"
export ANDROID_STL="gnustl_static"
# 强制使用gcc编译器, NDK r13已经默认使用clang编译器
export ANDROID_TOOLCHAIN="gcc"
# 使用NDK r13以上版本
export CMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"

######################## config ########################
# 使用protobuf-lite
export NCNN_USE_PROTOBUF_LITE="ON"

# 公用FLAGS
export COMMON_FLAGS=" "
# hidden symbol
if [ "$BUILD_HIDDEN_SYMBOL" != "OFF" ]; then
    COMMON_FLAGS="$COMMON_FLAGS -fvisibility=hidden -fvisibility-inlines-hidden"
fi
export BUILD_C_FLAGS="$BUILD_C_FLAGS $COMMON_FLAGS "
export BUILD_CXX_FLAGS="$BUILD_CXX_FLAGS $COMMON_FLAGS -frtti -fexceptions -std=c++11"

# 当前系统protoc可执行目录及信息
export THIRD_PARTY_ROOT=$current_dir/3rdparty
SYSTEM_PROTOBUF_PATH=$THIRD_PARTY_ROOT/libprotobuf/macOS
export PATH=$SYSTEM_PROTOBUF_PATH/bin:$PATH
export DYLD_LIBRARY_PATH=$SYSTEM_PROTOBUF_PATH/lib:$DYLD_LIBRARY_PATH
export PROTOBUF_PROTOC_EXE=$SYSTEM_PROTOBUF_PATH/bin/protoc

do_build()
{
    BUILD_ARCH=$1
    if [ "$BUILD_ARCH" == "armeabi-v7a" ]; then
        ANDROID_ABI="armeabi-v7a with NEON"
    else
        ANDROID_ABI="$BUILD_ARCH"
    fi

    # 3rdparty librarys
    export NEON2SSE_INCLUDE_DIRS=$THIRD_PARTY_ROOT/ARM_NEON_2_x86_SSE
    export NNPACK_INCLUDE_DIRS=$THIRD_PARTY_ROOT/libnnpack/Android/$BUILD_ARCH/include
    export NNPACK_LIBRARIES_DIRS=$THIRD_PARTY_ROOT/libnnpack/Android/$BUILD_ARCH/lib
    export EIGEN_ROOT=$THIRD_PARTY_ROOT/eigen
    export PROTOBUF_INCLUDE=$THIRD_PARTY_ROOT/libprotobuf/include
    if [ "$NCNN_USE_PROTOBUF_LITE" == "ON" ]; then
        export PROTOBUF_LIBRARY_STATIC=$THIRD_PARTY_ROOT/libprotobuf/android/$BUILD_ARCH/lib/libprotobuf-lite.a
        # 编译&安装目录
        export BUILD_DIR=$current_dir/libncnn/android-lite/build/$BUILD_ARCH
        export OUTPUT_DIR=$current_dir/libncnn/android-lite/$BUILD_ARCH

        export CAFFE_PROTO_INCLUDE_DIR=$current_dir/3rdparty/libcaffe-lite/android/$BUILD_ARCH/include
        export CAFFE_PROTO_LIB=$current_dir/3rdparty/libcaffe-lite/android/$BUILD_ARCH/lib/libcaffeproto.a
    else
        export PROTOBUF_LIBRARY_STATIC=$THIRD_PARTY_ROOT/libprotobuf/android/$BUILD_ARCH/lib/libprotobuf.a
        # 编译&安装目录
        export BUILD_DIR=$current_dir/libncnn/android/build/$BUILD_ARCH
        export OUTPUT_DIR=$current_dir/libncnn/android/$BUILD_ARCH

        export CAFFE_PROTO_INCLUDE_DIR=$current_dir/3rdparty/libcaffe/android/$BUILD_ARCH/include
        export CAFFE_PROTO_LIB=$current_dir/3rdparty/libcaffe/android/$BUILD_ARCH/lib/libcaffeproto.a
    fi

    # FLAGS
    export BUILD_COMMON_FLAGS="$COMMON_FLAGS -I$EIGEN_ROOT "
    export CMAKE_C_FLAGS="$BUILD_C_FLAGS $BUILD_COMMON_FLAGS"
    export CMAKE_CXX_FLAGS="$BUILD_CXX_FLAGS $BUILD_COMMON_FLAGS"

    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR
    pushd $BUILD_DIR

    cmake -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE           \
          -DANDROID_NDK="$ANDROID_NDK"                           \
          -DANDROID_TOOLCHAIN="$ANDROID_TOOLCHAIN"               \
          -DANDROID_ABI="$ANDROID_ABI"                           \
          -DANDROID_STL="$ANDROID_STL"                           \
          -DCMAKE_INSTALL_PREFIX="/"                             \
          -DCMAKE_BUILD_TYPE=$BUILD_TYPE                         \
          -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"               \
          -DCMAKE_C_FLAGS="$CMAKE_C_FLAGS"                       \
          -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS"                   \
          -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"               \
          -DPROTOBUF_INCLUDE_DIR=$PROTOBUF_INCLUDE               \
          -DPROTOBUF_LIBRARY=$PROTOBUF_LIBRARY_STATIC            \
          -DPROTOBUF_PROTOC_EXECUTABLE=$PROTOBUF_PROTOC_EXE      \
          -DNCNN_USE_PROTOBUF_LITE="$NCNN_USE_PROTOBUF_LITE"     \
          -DCAFFE_PROTO_INCLUDE_DIR="$CAFFE_PROTO_INCLUDE_DIR"   \
          -DCAFFE_PROTO_LIB="$CAFFE_PROTO_LIB"                   \
          $current_dir/..

    #cmake --build .
    make all -j8
    rm -rf $OUTPUT_DIR
    make install/strip DESTDIR=$OUTPUT_DIR

    popd
}

for ARCH in $ALL_ARCHS
do
    do_build $ARCH
done

popd # $current_dir
