#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/InferenceQueue.hpp"
#include "cpp_src/constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with Parametric Dueling Network support";

    py::class_<InferenceQueue>(m, "InferenceQueue")
        .def(py::init<>())
        .def("pop_all", &InferenceQueue::pop_all)
        .def("wait", &InferenceQueue::wait, py::call_guard<py::gil_scoped_release>());

    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def_readonly("infoset", &InferenceRequest::infoset)
        .def_readonly("num_actions", &InferenceRequest::num_actions)
        .def("set_result", [](InferenceRequest &req, std::vector<float> result) {
            req.promise.set_value(result);
        });

    // ИЗМЕНЕНО: Обновляем биндинги для SharedReplayBuffer
    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<uint64_t>(), py::arg("capacity"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count)
        .def("get_head", &ofc::SharedReplayBuffer::get_head)
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            // Создаем три numpy массива
            auto infosets_np = py::array_t<float>(batch_size * ofc::INFOSET_SIZE);
            auto actions_np = py::array_t<float>(batch_size * ofc::ACTION_VECTOR_SIZE);
            auto regrets_np = py::array_t<float>(batch_size);

            buffer.sample(
                batch_size, 
                static_cast<float*>(infosets_np.request().ptr), 
                static_cast<float*>(actions_np.request().ptr),
                static_cast<float*>(regrets_np.request().ptr)
            );

            // Решейпим
            infosets_np.resize({batch_size, ofc::INFOSET_SIZE});
            actions_np.resize({batch_size, ofc::ACTION_VECTOR_SIZE});
            regrets_np.resize({batch_size, 1}); // Для консистентности с MSELoss

            // Возвращаем кортеж из трех массивов
            return std::make_tuple(infosets_np, actions_np, regrets_np);
        }, py::arg("batch_size"));

    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<size_t, ofc::SharedReplayBuffer*, InferenceQueue*>(), 
             py::arg("action_limit"), py::arg("buffer"), py::arg("queue"))
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, py::call_guard<py::gil_scoped_release>());
}
