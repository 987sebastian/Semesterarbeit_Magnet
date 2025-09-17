cmake_minimum_required(VERSION 3.22)

project(biot_model LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23) # need > 17 because of std::numbers
add_library(biot_model STATIC src/model.cpp src/model_data.cpp)
target_include_directories(biot_model PUBLIC include)

add_executable(demo src/main.cpp)
target_link_libraries(demo PRIVATE biot_model)