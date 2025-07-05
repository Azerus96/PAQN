#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/InferenceQueue.hpp"
#include "cpp_src/constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with Policy-Value Network support";

    py::class_<PolicyRequestData>(m, "PolicyRequestData")
        .def_readonly("infoset", &PolicyRequestData::infoset)
        .def_readonly("action_vectors", &PolicyRequestData::action_vectors);

    py::class_<ValueRequestData>(m, "ValueRequestData")
        .def_readonly("infoset", &ValueRequestData::infoset);

    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def("get_type", [](InferenceRequest &req) {
            return req.data.index(); // 0 for Policy, 1 for Value
        })
        .def("get_policy_data", [](InferenceRequest &req) -> const PolicyRequestData& {
            return std::get<PolicyRequestData>(req.data);
        }, py::return_value_policy::reference_internal)
        .def("get_value_data", [](InferenceRequest &req) -> const ValueRequestData& {
            return std::get<ValueRequestData>(req.data);
        }, py::return_value_policy::reference_internal)
        .def("set_result", [](InferenceRequest &req, std::vector<float> result) {
            req.promise.set_value(result);
        });
        
    py::class_<InferenceQueue>(m, "InferenceQueue")
        .def(py::init<>())
        .def("pop_all", &InferenceQueue::pop_all)
        .def("wait", &InferenceQueue::wait, py::call_guard<py::gil_scoped_release>());

    py::class_<ofc::SharedReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<uint64_t>(), py::arg("capacity"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count)
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

    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<size_t, ofc::SharedReplayBuffer*, ofc::SharedReplayBuffer*, InferenceQueue*>(), 
             py::arg("action_limit"), py::arg("policy_buffer"), py::arg("value_buffer"), py::arg("queue"))
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, py::call_guard<py::gil_scoped_release>());
}
