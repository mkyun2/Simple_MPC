cmake_minimum_required(VERSION 3.14)
project(mpc_tutorial)

# C++ 표준 설정 (예: C++17)
#set(TARGET simulate)
 # 빌드 타겟
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/output)

find_package(Eigen3 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.16.0
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)    # 모든 대상 STATIC
set(BUILD_GTEST       ON  CACHE BOOL "" FORCE)
set(INSTALL_GTEST     OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_library(mpc_lib
    src/model.cpp
    src/mpc.cpp
)

# 헤더 디렉터리 (라이브러리 헤더를 외부에서 쓰려면 PUBLIC)
target_include_directories(mpc_lib
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_include_directories(mpc_lib PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(mpc_lib Eigen3::Eigen)

enable_testing()
# main을 빌드할 때 라이브러리 링크
#add_executable(${TARGET} main.cpp)
#target_link_libraries(${TARGET} PUBLIC mpc_lib)
#set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(Simulation
    main.cpp
)
add_executable(simulate ${Simulation})
target_link_libraries(simulate PRIVATE mpc_lib)


set(TestFiles
    test/test.cpp
)

add_executable(mpc_test ${TestFiles})
target_link_libraries(mpc_test PRIVATE mpc_lib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(mpc_test)
