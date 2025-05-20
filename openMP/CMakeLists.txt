cmake_minimum_required(VERSION 3.10)

project(openMP C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wpedantic -fopenmp -lm")

set(SRC_DIR src)
set(LIB_DIR include)

set(HEADERS
        ${LIB_DIR}/network.h
        include/utilities.h
        include/dataloader.h
        include/readconfig.h
        include/hpc.h
)

set(SOURCES
        ${SRC_DIR}/main.c
        src/utilities.c
        src/dataloader.c
        src/readconfig.c
)



add_executable(openMP ${SOURCES} ${HEADERS})

target_link_libraries(openMP m)

target_include_directories(openMP PRIVATE ${LIB_DIR})