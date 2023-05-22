#pragma once

#include <initializer_list>
#include <vector>
#include <memory>
#include "layer/layer.h"
#include "error-block/error_block.h"
#include "optimizers/gd/gd_optimizer.h"
#include "optimizers/sgd/sgd_optimizer.h"
#include "optimizers/adam/adam_optimizer.h"
#include "optimizers/sgd_momentum/sgd_momentum_optimizer.h"

namespace NeuralNetworkApp {

class NeuralNetwork {

public:
    NeuralNetwork(std::initializer_list<size_t> layers_sizes,
                  std::initializer_list<FunctionType> functions);

    template <typename T>
    NeuralNetwork(std::initializer_list<size_t> layers_sizes,
                  std::initializer_list<FunctionType> functions, T optimizer, ErrorType type) {
        CreateLayers(layers_sizes, functions);
        optimizer_ = std::make_unique<T>(std::move(optimizer));
        error_block_ = ErrorBlock(type);
    }

    void AddNextLayer(size_t input_size, size_t output_size, FunctionType func);

    template <typename T>
    void SetOptimizer(T optimizer) {
        optimizer_ = std::make_unique<T>(std::move(optimizer));
    }

    void SetError(ErrorType type);

    void Train(const std::vector<std::vector<double>>& train_input,
               const std::vector<std::vector<double>>& train_output, size_t max_iter_count);

    std::vector<double> Predict(const std::vector<double>& data) const;

private:
    std::vector<Layer> layers_;
    ErrorBlock error_block_ = ErrorBlock(ErrorType::MSE);
    std::unique_ptr<BaseOptimizer> optimizer_ = std::make_unique<AdamOptimizer>();

    void CreateLayers(const std::initializer_list<size_t>& layers_sizes,
                      const std::initializer_list<FunctionType>& functions);

    std::vector<Vector> TransformData(const std::vector<std::vector<double>>& data) const;
};

}  // namespace NeuralNetworkApp
