#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/constants.hpp"
#include "cpp_src/hand_evaluator.hpp"

namespace py = pybind11;

// Класс-менеджер для управления C++ потоками
class SolverManager {
public:
    SolverManager(
        size_t num_workers,
        size_t action_limit, 
        ofc::SharedReplayBuffer* policy_buffer, 
        ofc::SharedReplayBuffer* value_buffer,
        ofc::PolicyInferenceCallback policy_cb, 
        ofc::ValueInferenceCallback value_cb
    ) {
        stop_flag_.store(false);
        for (size_t i = 0; i < num_workers; ++i) {
            auto solver = std::make_unique<ofc::DeepMCCFR>(
                action_limit, policy_buffer, value_buffer, policy_cb, value_cb
            );
            threads_.emplace_back(&SolverManager::worker_loop, this, std::move(solver));
        }
    }

    ~SolverManager() {
        stop();
    }

    void stop() {
        if (!stop_flag_.exchange(true)) {
            for (auto& t : threads_) {
                if (t.joinable()) {
                    t.join();
                }
            }
        }
    }

private:
    void worker_loop(std::unique_ptr<ofc::DeepMCCFR> solver) {
        while (!stop_flag_.load()) {
            solver->run_traversal();
        }
    }

    std::vector<std::thread> threads_;
    std::atomic<bool> stop_flag_;
};

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with C++ Thread Manager";

    m.def("initialize_evaluator", []() {
        omp::HandEvaluator::initialize();
    }, py::call_guard<py::gil_scoped_release>());

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
        
    py::class_<SolverManager>(m, "SolverManager")
        .def(py::init<size_t, size_t, ofc::SharedReplayBuffer*, ofc::SharedReplayBuffer*, ofc::PolicyInferenceCallback, ofc::ValueInferenceCallback>(),
             py::arg("num_workers"), py::arg("action_limit"), py::arg("policy_buffer"), py::arg("value_buffer"),
             py::arg("policy_callback"), py::arg("value_callback"))
        .def("stop", &SolverManager::stop, py::call_guard<py::gil_scoped_release>());
}
