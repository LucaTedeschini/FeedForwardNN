cmake_minimum_required(VERSION 3.31) # È buona norma usare una versione recente
project(CUDA_Project LANGUAGES CXX CUDA) # Abilita C++ e CUDA

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 17) # Imposta anche lo standard C++ per coerenza

# Definisci le directory
set(SRC_DIR src)
set(INCLUDE_DIR include) # Nome più standard rispetto a LIB_DIR per gli header

# Raccogli i file sorgente
set (SOURCES
        ${SRC_DIR}/readconfig.cpp
        src/dataloader.cpp
)

# Aggiungi l'eseguibile con i suoi sorgenti
add_executable(CUDA main.cu ${SOURCES})

# Specifica le directory degli header per il tuo target
target_include_directories(CUDA PUBLIC ${INCLUDE_DIR}) # <-- QUESTA È LA RIGA CHIAVE

# Abilita la compilazione separabile
set_target_properties(CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)