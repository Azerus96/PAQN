#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
// #include <pybind11/variant.h> // УДАЛЕНО
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/InferenceQueue.hpp"
#include "cpp_src/constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with Policy-Value Network support";

    // --- Биндинги для структур данных запросов ---
    py::class_<ofc::PolicyRequestData>(m, "PolicyRequestData")
        .def_readonly("infoset", &ofc::PolicyRequestData::infoset)
        .def_readonly("action_vectors", &ofc::PolicyRequestData::action_vectors);

    py::class_<ofc::ValueRequestData>(m, "ValueRequestData")
        .def_readonly("infoset", &ofc::ValueRequestData::infoset);

    // --- Биндинг для enum ---
    py::enum_<ofc::RequestType>(m, "RequestType")
        .value("POLICY", ofc::RequestType::POLICY)
        .value("VALUE", ofc::RequestType::VALUE)
        .export_values();

    // --- Биндинг для универсального запроса ---
    py::class_<ofc::InferenceRequest>(m, "InferenceRequest")
        .def_readonly("type", &ofc::InferenceRequest::type)
        .def("get_policy_data", [](ofc::InferenceRequest &req) -> const ofc::PolicyRequestData& {
            if (req.type != ofc::RequestType::POLICY) throw std::runtime_error("Request is not of type POLICY");
            return req.data.policy_data;
        }, py::return_value_policy::reference_internal)
        .def("get_value_data", [](ofc::InferenceRequest &req) -> const ofc::ValueRequestData& {
            if (req.type != ofc::RequestType::VALUE) throw std::runtime_error("Request is not of type VALUE");
            return req.data.value_data;
        }, py::return_value_policy::reference_internal)
        .def("set_result", [](ofc::InferenceRequest &req, std::vector<float> result) {
            req.promise.set_value(result);
        });
        
    // --- Биндинг для очереди ---
    py::class_<ofc::InferenceQueue>(m, "InferenceQueue")
        .def(py::init<>())
        .def("pop_all", &ofc::InferenceQueue::pop_all)
        .def("wait", &ofc::InferenceQueue::wait, py::call_guard<py::gil_scoped_release>());

    // --- Биндинг для буфера воспроизведения ---
    py::class_<ofc::SharedReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<uint64_t>(), py::arg("capacity"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count)
        .def("get_head", &ofc::SharedReplayBuffer::get_head)
        .def("push", &ofc::SharedReplayBuffer::push, py::arg("infoset"), py::arg("action"), py::arg("target"))
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            auto infosets_np = py::array_t<float>(batch_size * ofc::INFOSET_SIZE);
            auto actions_np = py::array_t<float>(batch_size * ofc::ACTION_VECTOR_SIZE);
            auto targets_np = py::array_t<float>(batch_size);

            buffer.sample(
                batch_size, 
                static_cast<float*>(infosets_np.request().ptr), 
                static_cast<float*>(actions_np.request().ptr),
                static_cast<float*>(targets_np.request().ptr)
            );

            infosets_np.resize({batch_size, ofc::INFOSET_SIZE});
            actions_np.resize({batch_size, ofc::ACTION_VECTOR_SIZE});
            targets_np.resize({batch_size, 1});

            return std::make_tuple(infosets_np, actions_np, targets_np);
        }, py::arg("batch_size"));

    // --- Биндинг для основного класса DeepMCCFR ---
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<size_t, ofc::SharedReplayBuffer*, ofc::SharedReplayBuffer*, ofc::InferenceQueue*>(), 
             py::arg("action_limit"), py::arg("policy_buffer"), py::arg("value_buffer"), py::arg("queue"))
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, py::call_guard<py::gil_scoped_release>());
}
