# PAQN-main/CMakeLists.txt (ИСПРАВЛЕННАЯ ВЕРСИЯ 5.0 - ПУЛЕНЕПРОБИВАЕМАЯ)

cmake_minimum_required(VERSION 3.14) # Поднимаем версию для лучших практик
project(OFCEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Зависимости ---
include(FetchContent)

# Pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.1
)
FetchContent_MakeAvailable(pybind11)

# PyTorch
find_package(Torch REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)


# --- Исходные файлы ---
set(SOURCES
    "cpp_src/DeepMCCFR.cpp"
    "cpp_src/game_state.cpp"
    "cpp_src/ompeval/omp/CardRange.cpp"
    "cpp_src/ompeval/omp/CombinedRange.cpp"
    "cpp_src/ompeval/omp/EquityCalculator.cpp"
    "cpp_src/ompeval/omp/HandEvaluator.cpp"
    "pybind_wrapper.cpp"
)

# --- Создание цели (Python модуль) ---
# Используем pybind11_add_module для создания модуля
# SHARED - означает, что мы создаем .so файл
pybind11_add_module(ofc_engine SHARED ${SOURCES})


# --- Настройка путей и линковки ---

# 1. Добавляем пути к нашим собственным заголовочным файлам
target_include_directories(ofc_engine PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp_src"
    "${PROJECT_SOURCE_DIR}/cpp_src/ompeval"
    "${PROJECT_SOURCE_DIR}/cpp_src/concurrentqueue"
)

# 2. Линкуем зависимости
# Это САМЫЙ ВАЖНЫЙ БЛОК.
# Мы линкуем ofc_engine с целями, которые предоставляют зависимости.
# CMake автоматически разрешит все пути к заголовочным файлам и библиотекам.
target_link_libraries(ofc_engine PRIVATE
    # Линковка с pybind11::module автоматически подтягивает pybind11::headers
    # и решает проблему с "variant.h not found".
    pybind11::module
    
    # Линковка с библиотеками PyTorch
    ${TORCH_LIBRARIES}
    
    # Линковка с OpenMP
    OpenMP::OpenMP_CXX
)

# 3. Опции компиляции
target_compile_options(ofc_engine PRIVATE -O3 -g)


# --- Отладочные сообщения (опционально, но полезно) ---
message(STATUS "Pybind11 version: ${pybind11_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
