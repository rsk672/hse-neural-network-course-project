#pragma once

#include <initializer_list>
#include <vector>
#include <memory>
#include "layer.h"
#include "error_block.h"
#include "optimizers/gd_optimizer.h"
#include "optimizers/sgd_optimizer.h"
#include "optimizers/optimizer_creator.h"
#include "optimizers/sag_optimizer.h"
#include "optimizers/adam_optimizer.h"

namespace NeuralNetworkApp {

class NeuralNetwork {

public:
    NeuralNetwork(std::initializer_list<size_t> layers_sizes,
                  std::initializer_list<FunctionType> functions);

    void AddNextLayer(size_t input_size, size_t output_size, FunctionType func);

    template <typename... Args>
    void SetOptimizer(OptimizerType type, Args&&... args) {
        switch (type) {
            case GD:
                optimizer_ = OptimizerCreator::Create<GDOptimizer>(std::forward<Args>(args)...);
                break;
            case SGD:
                optimizer_ = OptimizerCreator::Create<SGDOptimizer>(std::forward<Args>(args)...);
                break;
            case SAG:
                optimizer_ = OptimizerCreator::Create<SAGOptimizer>(std::forward<Args>(args)...);
                break;
            case Adam:
                optimizer_ = OptimizerCreator::Create<AdamOptimizer>(std::forward<Args>(args)...);
                break;
            default:
                assert(false);
                optimizer_ = std::make_unique<GDOptimizer>();
        }
    }

    void SetError(ErrorType type);

    void Train(const std::vector<std::vector<double>>& train_input,
               const std::vector<std::vector<double>>& train_output, double error,
               size_t max_iter_count);

    std::vector<double> Predict(const std::vector<double>& data);

private:
    std::vector<Layer> layers_;
    ErrorBlock error_block_ = ErrorBlock(ErrorType::MSE);
    std::unique_ptr<Optimizer> optimizer_ = std::make_unique<GDOptimizer>();
};

}  // namespace NeuralNetworkApp
