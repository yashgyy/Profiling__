#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

int main() {
    try {
        // Load the traced model
        torch::jit::script::Module net = torch::jit::load("../net_traced.pt");
        
        // Create multi-threaded data loader for MNIST data
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(torch::data::datasets::MNIST("../data").map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
                    torch::data::transforms::Stack<>())), 64);
        
        // Set to training mode
        net.train();
        
        // Get all parameters from the module for the optimizer
        std::vector<torch::Tensor> parameters;
        for (const auto& p : net.parameters()) {
            parameters.push_back(p);
        }
        
        // Create optimizer
        torch::optim::SGD optimizer(parameters, /*lr=*/0.01);
        
        // Training loop
        for(size_t epoch=1; epoch<=10; ++epoch) {
            size_t batch_index = 0;
            
            for (auto& batch : *data_loader) {
                // Reset gradients
                optimizer.zero_grad();
                
                // Prepare input
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(batch.data);
                
                // Forward pass
                torch::Tensor prediction = net.forward(inputs).toTensor();
                
                // Compute loss
                torch::Tensor loss = torch::nll_loss(prediction, batch.target);
                
                // Backward pass
                loss.backward();
                
                // Update parameters
                optimizer.step();
                
                // Output the loss and checkpoint every 100 batches
                if (++batch_index % 100 == 0) {
                    std::cout << "Continue Training - Epoch: " << epoch << " | Batch: " << batch_index 
                        << " | Loss: " << loss.item<float>() << std::endl;
                    
                    // Save the updated model
                    net.save("net_traced_continued.pt");
                }
            }
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading/running the model: " << e.msg() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}