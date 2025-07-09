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
        size_t action_limit,
        ofc::SharedReplayBuffer* policy_buffer, 
        ofc::SharedReplayBuffer* value_buffer,
        py::object request_queue, // Принимаем по значению
        py::dict result_queue     // Принимаем по значению
    ) : request_queue_(request_queue), result_queue_(result_queue) { // Сохраняем копии
        stop_flag_.store(false);
        for (size_t i = 0; i < num_workers; ++i) {
            // Передаем сохраненные копии в конструктор DeepMCCFR
            auto solver = std::make_unique<ofc::DeepMCCFR>(
                action_limit, policy_buffer, value_buffer, &request_queue_, &result_queue_
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
    // Храним объекты pybind здесь, чтобы они были живы, пока работают потоки
    py::object request_queue_;
    py::dict result_queue_;
};

class PySolverManager {
public:
    PySolverManager(
        size_t num_workers,
        size_t action_limit,
        ofc::SharedReplayBuffer* policy_buffer, 
        ofc::SharedReplayBuffer* value_buffer,
        py::object request_queue,
        py::dict result_dict
    ) {
        py::gil_scoped_release release;
        // Просто передаем объекты дальше
        impl_ = std::make_unique<SolverManagerImpl>(
            num_workers, action_limit, policy_buffer, value_buffer,
            request_queue, result_dict
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
    std::unique_ptr<SolverManagerImpl> impl_;
};


PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with C++ Thread Manager and Queue-based Inference";

    m.def("initialize_evaluator", []() {
        omp::HandEvaluator::initialize();
    }, "Initializes the static hand evaluator lookup tables.", py::call_guard<py::gil_scoped_release>());

    py::class_<ofc::SharedReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<uint64_t>(), py::arg("capacity"))
        .def("size", &ofc::SharedReplayBuffer::size)
        .def("total_generated", &ofc::SharedReplayBuffer::total_generated)
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) -> py::object {
            auto infosets_np = py::array_t<float>({batch_size, ofc::INFOSET_SIZE});
            auto actions_np = py::array_t<float>({batch_size, ofc::ACTION_VECTOR_SIZE});
            auto targets_np = py::array_t<float>({batch_size});
            
            bool success = buffer.sample(
                batch_size, 
                infosets_np,
                actions_np,
                targets_np
            );

            if (!success) {
                return py::none();
            }
            
            return py::cast(std::make_tuple(infosets_np, actions_np, targets_np));
        }, py::arg("batch_size"), "Samples a batch from the buffer.");
        
    py::class_<PySolverManager>(m, "SolverManager")
        .def(py::init<size_t, size_t, ofc::SharedReplayBuffer*, ofc::SharedReplayBuffer*, py::object, py::dict>(),
             py::arg("num_workers"),
             py::arg("action_limit"),
             py::arg("policy_buffer"), py::arg("value_buffer"),
             py::arg("request_queue"), py::arg("result_queue"))
        .def("stop", &PySolverManager::stop);
}
