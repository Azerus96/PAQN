cmake_minimum_required(VERSION 3.14)
project(OFCEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Зависимости ---
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
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
pybind11_add_module(ofc_engine SHARED ${SOURCES})


# --- Настройка путей и линковки ---
target_include_directories(ofc_engine PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp_src"
    "${PROJECT_SOURCE_DIR}/cpp_src/ompeval"
    "${PROJECT_SOURCE_DIR}/cpp_src/concurrentqueue"
)

target_link_libraries(ofc_engine PRIVATE
    pybind11::module
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# --- Опции компиляции ---
# ИЗМЕНЕНО: Добавлен флаг для perf
target_compile_options(ofc_engine PRIVATE -O3 -g -fno-omit-frame-pointer)


# --- Отладочные сообщения ---
message(STATUS "Pybind11 version: ${pybind11_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
