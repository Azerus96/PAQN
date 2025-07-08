#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <iostream>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/constants.hpp"
#include "cpp_src/hand_evaluator.hpp"
#include "cpp_src/InferenceQueue.hpp"

namespace py = pybind11;

class SolverManagerImpl {
public:
    SolverManagerImpl(
        size_t num_workers,
        ofc::SharedReplayBuffer* policy_buffer, 
        ofc::SharedReplayBuffer* value_buffer,
        py::object* request_queue,
        py::object* result_queue
    ) {
        stop_flag_.store(false);
        for (size_t i = 0; i < num_workers; ++i) {
            auto solver = std::make_unique<ofc::DeepMCCFR>(
                policy_buffer, value_buffer, request_queue, result_queue
            );
            threads_.emplace_back(&SolverManagerImpl::worker_loop, this, std::move(solver));
        }
    }

    ~SolverManagerImpl() {
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

class PySolverManager {
public:
    PySolverManager(
        size_t num_workers,
        ofc::SharedReplayBuffer* policy_buffer, 
        ofc::SharedReplayBuffer* value_buffer,
        py::object request_queue,
        py::object result_queue
    ) : request_queue_(request_queue), result_queue_(result_queue) {
        py::gil_scoped_release release;
        impl_ = std::make_unique<SolverManagerImpl>(
            num_workers, policy_buffer, value_buffer,
            &request_queue_, &result_queue_
        );
    }

    ~PySolverManager() {
        py::gil_scoped_acquire acquire;
        if (impl_) {
            impl_->stop();
        }
    }

    void stop() {
        py::gil_scoped_release release;
        if (impl_) {
            impl_->stop();
        }
    }

private:
    py::object request_queue_;
    py::object result_queue_;
    std::unique_ptr<SolverManagerImpl> impl_;
};


PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with C++ Thread Manager and Queue-based Inference";

    py::class_<ofc::InferenceRequest>(m, "InferenceRequest")
        .def(py::init<>())
        .def_readwrite("id", &ofc::InferenceRequest::id)
        .def_readwrite("is_policy_request", &ofc::InferenceRequest::is_policy_request)
        .def_readwrite("raw_state", &ofc::InferenceRequest::raw_state)
        .def_readwrite("action_vectors", &ofc::InferenceRequest::action_vectors);

    py::class_<ofc::InferenceResult>(m, "InferenceResult")
        .def(py::init<>())
        .def_readwrite("id", &ofc::InferenceResult::id)
        .def_readwrite("is_policy_result", &ofc::InferenceResult::is_policy_result)
        .def_readwrite("predictions", &ofc::InferenceResult::predictions);

    m.def("initialize_evaluator", []() {
        omp::HandEvaluator::initialize();
    }, "Initializes the static hand evaluator lookup tables.", py::call_guard<py::gil_scoped_release>());

    py::class_<ofc::SharedReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<uint64_t>(), py::arg("capacity"))
        .def("size", &ofc::SharedReplayBuffer::size)
        .def("total_generated", &ofc::SharedReplayBuffer::total_generated)
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) -> py::object {
            std::vector<std::vector<int>> raw_states;
            auto actions_np = py::array_t<float>({batch_size, ofc::ACTION_VECTOR_SIZE});
            auto targets_np = py::array_t<float>({batch_size}); // ИСПРАВЛЕНИЕ: Цель - скаляр, а не вектор
            
            bool success = buffer.sample(
                batch_size, 
                raw_states,
                actions_np,
                targets_np
            );

            if (!success) {
                return py::none();
            }
            
            return py::cast(std::make_tuple(py::cast(raw_states), actions_np, targets_np));
        }, py::arg("batch_size"), "Samples a batch from the buffer.");
        
    py::class_<PySolverManager>(m, "SolverManager")
        .def(py::init<size_t, ofc::SharedReplayBuffer*, ofc::SharedReplayBuffer*, py::object, py::object>(),
             py::arg("num_workers"),
             py::arg("policy_buffer"), py::arg("value_buffer"),
             py::arg("request_queue"), py::arg("result_queue"))
        .def("stop", &PySolverManager::stop);
}
