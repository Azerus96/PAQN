cmake_minimum_required(VERSION 3.14)
project(OFCEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Зависимости ---
# find_package(pybind11 REQUIRED) # Временно отключаем для чистого C++ теста
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED) # Добавляем для std::thread

# --- Исходные файлы ---
set(SOURCES
    "cpp_src/DeepMCCFR.cpp"
    "cpp_src/game_state.cpp"
    "cpp_src/ompeval/omp/CardRange.cpp"
    "cpp_src/ompeval/omp/CombinedRange.cpp"
    "cpp_src/ompeval/omp/EquityCalculator.cpp"
    "cpp_src/ompeval/omp/HandEvaluator.cpp"
    # "pybind_wrapper.cpp" # Временно отключаем
)

# --- Создание цели (Python модуль) ---
# pybind11_add_module(ofc_engine SHARED ${SOURCES}) # Временно отключаем

# --- НОВАЯ ЦЕЛЬ: ЧИСТЫЙ C++ ТЕСТ ---
add_executable(main_test cpp_src/main_test.cpp ${SOURCES})

# --- Настройка путей и линковки ---
target_include_directories(main_test PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp_src"
    "${PROJECT_SOURCE_DIR}/cpp_src/ompeval"
    "${PROJECT_SOURCE_DIR}/cpp_src/concurrentqueue"
)

target_link_libraries(main_test PRIVATE
    # pybind11::module # Не нужно для C++ теста
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# --- Опции компиляции ---
target_compile_options(main_test PRIVATE -O3 -g -fno-omit-frame-pointer)

# --- Отладочные сообщения ---
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
