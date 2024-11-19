# CMakeLists.txt
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fedtemp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Torch configuration
set(TORCH_DIR "" CACHE PATH "Path to LibTorch installation")

# Boost configuration
set(BOOST_DIR "" CACHE PATH "Path to Boost installation")

# Find dependencies
find_package(Torch REQUIRED PATHS ${TORCH_DIR})
find_package(Boost REQUIRED PATHS ${BOOST_DIR} COMPONENTS system thread)

# Set C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Source files
file(GLOB_RECURSE CLIENT_SOURCE_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/client/*.cpp)
file(GLOB_RECURSE SERVER_SOURCE_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/server/*.cpp)

# Client executable
add_executable(client ${CLIENT_SOURCE_FILES})
target_include_directories(client PUBLIC 
    ${PROJECT_SOURCE_DIR}/client 
    ${TORCH_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS}
)
target_link_directories(client PUBLIC 
    ${Torch_LIBDIR} 
    ${Boost_LIBRARY_DIRS}
)
target_link_libraries(client ${TORCH_LIBRARIES} Boost::system Boost::thread)

# Server executable
add_executable(server ${SERVER_SOURCE_FILES})
target_include_directories(server PUBLIC 
    ${PROJECT_SOURCE_DIR}/server 
    ${TORCH_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS}
)
target_link_directories(server PUBLIC 
    ${Torch_LIBDIR} 
    ${Boost_LIBRARY_DIRS}
)
target_link_libraries(server ${TORCH_LIBRARIES} Boost::system Boost::thread)

# Target properties
set_target_properties(client PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/client"
)
set_target_properties(server PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/server"
)
