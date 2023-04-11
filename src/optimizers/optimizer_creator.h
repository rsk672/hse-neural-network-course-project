#pragma once
#include <memory>
#include <type_traits>
#include "gd_optimizer.h"
#include "optimizer.h"

namespace NeuralNetworkApp {

class OptimizerCreator {

public:
    template <typename T, typename... Args>
    static std::enable_if_t<std::is_constructible<T, Args...>::value, std::unique_ptr<Optimizer>>
    Create(Args&&... args) {
        return std::make_unique<T>(std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    static std::enable_if_t<!std::is_constructible<T, Args...>::value, std::unique_ptr<Optimizer>>
    Create(Args... args) {
        return nullptr;
    }
};

}  // namespace NeuralNetworkApp
