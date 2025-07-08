#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <variant>
#include <vector>
#include <string>
#include <cstdint>
#include "constants.hpp"

namespace py = pybind11;

namespace ofc {

// Структура запроса от C++ к Python
struct InferenceRequest {
    uint64_t id;
    bool is_policy_request;
    // --- ИЗМЕНЕНИЕ ---: Вместо вектора float для инфосета, передаем вектор int для сырого состояния
    std::vector<int> raw_state; 
    // Для policy-запроса, содержит векторы действий. Для value - пустой.
    std::vector<std::vector<float>> action_vectors; 
};

// Структура ответа от Python к C++
struct InferenceResult {
    uint64_t id;
    bool is_policy_result;
    // Для policy-ответа, содержит логиты. Для value - один элемент (сама ценность).
    std::vector<float> predictions; 
};

// Очередь запросов (C++ -> Python)
using InferenceRequestQueue = py::object;

// Очередь результатов (Python -> C++)
using InferenceResultQueue = py::object;

} // namespace ofc
