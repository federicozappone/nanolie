cmake_minimum_required(VERSION 3.10)
project(nanolie)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_LIST_DIR}/build/compile_commands.json ${CMAKE_CURRENT_LIST_DIR}/compile_commands.json
)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

set(EIGEN_BUILD_DOC OFF)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)

FetchContent_MakeAvailable(Eigen3)

include_directories(${Eigen3_SOURCE_DIR})

enable_testing()

add_executable(utest_so3 utest_so3.cc)
target_link_libraries(utest_so3 GTest::gtest_main)

add_executable(utest_se3 utest_se3.cc)
target_link_libraries(utest_se3 GTest::gtest_main)


include(GoogleTest)
gtest_discover_tests(utest_so3)
gtest_discover_tests(utest_se3)
