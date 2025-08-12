#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <iostream>
#include <map>
#include <mutex>
#include <condition_variable>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/constants.hpp"
#include "cpp_src/hand_evaluator.hpp"
#include "cpp_src/InferenceQueue.hpp"

namespace py = pybind11;

// Потокобезопасный C++ мост для ожидания результатов
class ResultBridge {
public:
    void add_result(uint64_t id, py::tuple result) {
        std::lock_guard<std::mutex> lock(mtx_);
        results_[id] = result;
        cv_.notify_all();
    }

    py::tuple get_result(uint64_t id) {
        std::unique_lock<std::mutex> lock(mtx_);
        // Освобождаем GIL на время ожидания результата
        cv_.wait(lock, py::gil_scoped_release(), [this, id] { return results_.count(id); });
        py::tuple result = results_[id];
        results_.erase(id);
        return result;
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::map<uint64_t, py::tuple> results_;
};

class SolverManager {
public:
    SolverManager(
        size_t num_workers,
        size_t action_limit,
        ofc::SharedReplayBuffer* policy_buffer, 
        ofc::SharedReplayBuffer* value_buffer,
        py::object request_queue
    ) : num_workers_(num_workers),
        action_limit_(action_limit),
        policy_buffer_(policy_buffer),
        value_buffer_(value_buffer),
        request_queue_(request_queue)
    {
        stop_flag_.store(true);
    }

    ~SolverManager() {
        stop();
    }

    void start() {
        if (!stop_flag_.exchange(false)) {
            return;
        }

        result_bridge_ = std::make_shared<ResultBridge>();

        for (size_t i = 0; i < num_workers_; ++i) {
            auto request_cb = [this](uint64_t id, bool is_policy, const std::vector<float>& infoset, const std::vector<std::vector<float>>& actions) {
                // Этот коллбэк вызывается с захваченным GIL
                py::tuple request_tuple = py::make_tuple(id, is_policy, py::cast(infoset), py::cast(actions));
                this->request_queue_.attr("put")(request_tuple);
            };

            auto result_cb = [this](uint64_t id) -> py::tuple {
                // Этот коллбэк вызывается с захваченным GIL
                return this->result_bridge_->get_result(id);
            };

            auto solver = std::make_unique<ofc::DeepMCCFR>(
                action_limit_, policy_buffer_, value_buffer_, request_cb, result_cb
            );
            threads_.emplace_back(&SolverManager::worker_loop, this, std::move(solver));
        }
    }

    void stop() {
        if (!stop_flag_.exchange(true)) {
            for (auto& t : threads_) {
                if (t.joinable()) {
                    t.join();
                }
            }
            threads_.clear();
        }
    }

    void add_result(uint64_t id, py::tuple result) {
        if (result_bridge_) {
            result_bridge_->add_result(id, result);
        }
    }

private:
    void worker_loop(std::unique_ptr<ofc::DeepMCCFR> solver) {
        // Захватываем GIL один раз для всего жизненного цикла потока
        py::gil_scoped_acquire acquire;
        while (!stop_flag_.load()) {
            solver->run_traversal();
        }
    }

    size_t num_workers_;
    size_t action_limit_;
    ofc::SharedReplayBuffer* policy_buffer_;
    ofc::SharedReplayBuffer* value_buffer_;

    std::vector<std::thread> threads_;
    std::atomic<bool> stop_flag_;
    
    py::object request_queue_;
    std::shared_ptr<ResultBridge> result_bridge_;
};


PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with C++ Thread Manager and callback-based Inference";

    m.def("initialize_evaluator", []() {
        omp::HandEvaluator::initialize();
    }, "Initializes the static hand evaluator lookup tables.", py::call_guard<py::gil_scoped_release>());

    py::class_<ofc::SharedReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<uint64_t>(), py::arg("capacity"))
        .def("size", &ofc::SharedReplayBuffer::size)
        .def("total_generated", &ofc::SharedReplayBuffer::total_generated)
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) -> py::object {
            py::gil_scoped_release release; // Сэмплирование - это C++ операция
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
        
    py::class_<SolverManager>(m, "SolverManager")
        .def(py::init<size_t, size_t, ofc::SharedReplayBuffer*, ofc::SharedReplayBuffer*, py::object>(),
             py::arg("num_workers"),
             py::arg("action_limit"),
             py::arg("policy_buffer"), py::arg("value_buffer"),
             py::arg("request_queue")
        )
        .def("start", &SolverManager::start)
        .def("stop", &SolverManager::stop)
        .def("add_result", &SolverManager::add_result);
}
