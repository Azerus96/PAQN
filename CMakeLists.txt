# D2CFR-main/CMakeLists.txt (ИСПРАВЛЕННАЯ ВЕРСИЯ 4.0 - ФИНАЛЬНАЯ)

cmake_minimum_required(VERSION 3.10)
project(OFCEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.1
)
FetchContent_MakeAvailable(pybind11)

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# Явно указываем ВСЕ необходимые исходные файлы.
set(SOURCES
    "cpp_src/DeepMCCFR.cpp"
    "cpp_src/game_state.cpp"
    "cpp_src/ompeval/omp/CardRange.cpp"
    "cpp_src/ompeval/omp/CombinedRange.cpp"
    "cpp_src/ompeval/omp/EquityCalculator.cpp"
    "cpp_src/ompeval/omp/HandEvaluator.cpp"
    "pybind_wrapper.cpp"
)

pybind11_add_module(ofc_engine SHARED ${SOURCES})

# Указываем пути к нашим собственным заголовочным файлам
target_include_directories(ofc_engine PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp_src"
    "${PROJECT_SOURCE_DIR}/cpp_src/ompeval"
    "${PROJECT_SOURCE_DIR}/cpp_src/concurrentqueue"
)

# ИЗМЕНЕНО: Линкуем не только библиотеки, но и заголовочные файлы pybind11.
# Это самый надежный способ передать компилятору все пути, включая те,
# что нужны для <pybind11/variant.h>, <pybind11/stl.h> и т.д.
target_link_libraries(ofc_engine PRIVATE 
    pybind11::headers       # <-- ДОБАВЛЕНО
    OpenMP::OpenMP_CXX 
    ${TORCH_LIBRARIES}
)

target_compile_options(ofc_engine PRIVATE -O3 -g)

# Сообщения для отладки можно оставить, они полезны
message(STATUS "Pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
