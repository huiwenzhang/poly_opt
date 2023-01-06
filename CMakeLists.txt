cmake_minimum_required(VERSION 3.5)
set(CMAKE_BUILD_TYPE "Debug")
project(poly_opt)

set(CMAKE_CXX_STANDARD 14)

add_definitions("-g")

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(
    include
)


file(GLOB_RECURSE SRC_BCD ${PROJECT_SOURCE_DIR}/src/*cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_BCD})

target_link_libraries(${PROJECT_NAME} 
    nlopt
    ${Boost_LIBRARIES}
)

add_executable(test_poly_linear test/test_linear_opt.cpp)
target_link_libraries(test_poly_linear ${PROJECT_NAME})

add_executable(generate_car_path test/b11_car_path.cpp)
target_link_libraries(generate_car_path ${PROJECT_NAME})

add_executable(test_poly_nonlinear test/test_nonlinear_opt.cpp)
target_link_libraries(test_poly_nonlinear ${PROJECT_NAME})