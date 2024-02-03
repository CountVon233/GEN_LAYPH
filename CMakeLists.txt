cmake_minimum_required(VERSION 3.1)

project(
  Gen_layph
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(run_genlayph src/genlayph.cpp)
add_executable(run_testpr src/test-pagerank.cpp)
add_executable(run_statistics src/statistics.cpp)
add_executable(run_filter_louvain src/pre-filter-comm.cpp)

target_link_libraries(run_genlayph PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(run_filter_louvain PRIVATE OpenMP::OpenMP_CXX)