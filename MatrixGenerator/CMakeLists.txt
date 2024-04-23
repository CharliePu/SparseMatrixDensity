cmake_minimum_required(VERSION 3.12)

project(SparseMatrixDensity LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# Boost python package requirement
find_package(Boost COMPONENTS system python3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# MatrixGenerator
add_library(MatrixGenerator MODULE src/MatrixGenerator.cpp)
target_link_libraries(MatrixGenerator PUBLIC ${Boost_LIBRARIES} ${Python3_LIBRARIES})

# Include directories
target_include_directories(MatrixGenerator PRIVATE 
${CMAKE_SOURCE_DIR}/include
${Boost_INCLUDE_DIRS}
${Python3_INCLUDE_DIRS}
)

# Boost python lib output
set_target_properties(MatrixGenerator PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ../MatrixGenerator/lib
)
set_target_properties(MatrixGenerator PROPERTIES PREFIX "")
set_target_properties(MatrixGenerator PROPERTIES OUTPUT_NAME "MatrixGenerator")
