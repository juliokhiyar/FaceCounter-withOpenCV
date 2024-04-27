cmake_minimum_required(VERSION 3.12)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Set the source files for your project
set(SOURCE_FILES tutorial.cpp)

# Create the executable
add_executable(MyProject ${SOURCE_FILES})

# Link against OpenCV
target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS})
