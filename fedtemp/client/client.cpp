#include "FedComm.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

int main()
{
	std::string server_ip, port, data_dir, model_name;
	
	std::cout << "Enter server ip: "; 
	std::cin >> server_ip;
	
	std::cout << "Enter server port: "; 
	std::cin >> port;
	
	std::cout << "Enter model name: "; 
	std::cin >> model_name;
	
	FedComm fedcomm {server_ip, port};
	
    try
	{
        torch::jit::script::Module net = fedcomm.get_model(model_name);
		
		std::cout << "Enter data directory: ";
		std::cin >> data_dir;
		
        // Create multi-threaded data loader for MNIST data
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(
				torch::data::datasets::MNIST(data_dir).map(torch::data::transforms::Normalize<>(0.13707, 0.3081))
				.map(torch::data::transforms::Stack<>())
			), 64
		);
        
        // Set to training mode
        net.train();
        
        // Get all parameters from the module for the optimizer
        std::vector<torch::Tensor> parameters;
        for (const auto& param : net.parameters()) parameters.push_back(param);
        
        // Create optimizer
        torch::optim::SGD optimizer(parameters, /*lr=*/0.01);
        
        // Training loop
        for(size_t epoch=1; epoch<=10; ++epoch)
		{
            size_t batch_index = 0;
            
            for (auto& batch : *data_loader)
			{
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
                    // net.save(model_file_path);
                }
            }
			
			fedcomm.submit_params(model_name, net);
			
			std::vector<torch::Tensor> received_params = fedcomm.get_params(model_name);
			
			std::size_t param_count = 0;
			auto net_params = net.parameters();
			for(auto& param : net_params) param = received_params[param_count++];
			
			net_params = net.parameters();
			for(int i=0; i<received_params.size(); i++)
			{
				if(net_params[i]!=received_params[i]) throw std::runtime_error("Model params updation not working.\n");
			}
        }
    }
    catch (const c10::Error& e)
	{
        std::cerr << "c10 Error: " << e.msg() << std::endl;
    }
    catch (const std::exception& e)
	{
        std::cerr << "Standard error: " << e.what() << std::endl;
    }
	
    fedcomm.close(model_name);
    return 0;
}
