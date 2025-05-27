#pragma once

#include "hetu/common/macros.h"
#include <type_traits>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>

namespace hetu {

// A lightweight task queue
class TaskQueue final {
  using Task = std::tuple<std::function<void()>, uint64_t, std::string>;

 public:
  TaskQueue(const std::string& queue_name, size_t num_workers,
            uint64_t max_pending_tasks = 10240UL)
  : _queue_name(queue_name), _max_pending_tasks(max_pending_tasks) {
    HT_ASSERT(num_workers > 0) << "Number of workers must be positive.";
    HT_ASSERT(max_pending_tasks > 0) << "Max pending tasks must be positive.";
    _workers.reserve(num_workers);
    for (size_t i = 0; i < num_workers; i++)
      _workers.emplace_back(&TaskQueue::_RunWorker, this, i);
    _alive = 1;
    _shutdowned = false;
  }

  ~TaskQueue() {
    if (!_shutdowned)
      Shutdown();
    for (auto& worker : _workers)
      worker.join();
  }

  std::future<void> Enqueue(std::function<void()> f,
                            const std::string& name = "") {
    HT_ASSERT(!_shutdowned) << "The task queue has been shutdowned.";

    auto task_f = std::make_shared<std::packaged_task<void()>>(std::move(f));
    auto future = task_f->get_future();

    {
      std::unique_lock<std::mutex> lock(_mutex);
      while (_tasks.size() >= _max_pending_tasks) {
        _dequeue_signal.wait(lock);
        HT_ASSERT(!_shutdowned) << "The task queue has been shutdowned.";
      }
      auto task_id = _num_enqueued_tasks++;
      _tasks.push_back({[task_f] { (*task_f)(); }, task_id, name});
      _enqueue_signal.notify_one();
    }

    return future;
  }

  void Shutdown() {
    std::unique_lock<std::mutex> lock(_mutex);
    _shutdowned = true;
    _alive = 0;
    _enqueue_signal.notify_all();
    _dequeue_signal.notify_all();
  }

  const std::string& name() const {
    return _queue_name;
  }

  uint64_t max_pending_tasks() const {
    return _max_pending_tasks;
  }

  uint64_t num_enqueued_tasks() const {
    return _num_enqueued_tasks;
  }

  int num_workers() const {
    return _workers.size();
  }

  bool running() const {
    return _alive == 1;
  }
  
  int alive() const {
    return _alive;
  }

 private:
  static void _RunWorker(TaskQueue* const task_queue, int worker_id) {
    uint64_t num_processed = 0;
    const std::string worker_name = "TaskQueue[" + task_queue->name() +
      "] Worker[" + std::to_string(worker_id) + "]";
    while (true) {
      std::unique_lock<std::mutex> lock(task_queue->_mutex);
      if (task_queue->_tasks.empty()) {
        task_queue->_enqueue_signal.wait(lock, [&task_queue] {
          return task_queue->_shutdowned || !task_queue->_tasks.empty();
        });
        if (task_queue->_shutdowned && task_queue->_tasks.empty())
          break;
      } else {
        auto task = std::move(task_queue->_tasks.front());
        task_queue->_tasks.pop_front();
        task_queue->_dequeue_signal.notify_one();
        lock.unlock();

        auto& task_id = std::get<1>(task);
        auto& task_name = std::get<2>(task);
        HT_LOG_TRACE << worker_name << " Processing task " << task_id << " \""
                     << task_name << "\"...";
        std::get<0>(task)();
        HT_LOG_TRACE << worker_name << " Processed task " << task_id << " \""
                     << task_name << "\" successfully.";
        num_processed++;
      }
    }
    if (num_processed > 0)
      HT_LOG_DEBUG << worker_name << " Summary: " << num_processed
                   << " processed task(s).";
  }

  const std::string _queue_name;
  uint64_t _max_pending_tasks;
  std::atomic<uint64_t> _num_enqueued_tasks{0};
  std::deque<Task> _tasks;

  std::mutex _mutex;
  std::condition_variable _enqueue_signal;
  std::condition_variable _dequeue_signal;
  std::vector<std::thread> _workers;
  bool _shutdowned;
  int _alive;
};

} // namespace hetu
