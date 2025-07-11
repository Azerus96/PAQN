#pragma once

// --- ИСПРАВЛЕНИЕ: Добавляем базовые заголовки ДО pybind11 ---
#include <cstddef>
#include <vector>
#include <string>
#include <cstdint>
// --- КОНЕЦ ИСПРАВЛЕНИЯ ---

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <variant>
#include "constants.hpp"

namespace py = pybind11;

namespace ofc {

struct InferenceRequest {
    uint64_t id;
    bool is_policy_request;
    std::vector<float> infoset; 
    std::vector<std::vector<float>> action_vectors; 
};

struct InferenceResult {
    uint64_t id;
    bool is_policy_result;
    std::vector<float> predictions; 
};

using InferenceRequestQueue = py::object;
using InferenceResultQueue = py::object;

} // namespace ofc
