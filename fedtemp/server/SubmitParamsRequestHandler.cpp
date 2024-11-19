#include "SubmitParamsRequestHandler.h"

namespace
{
	std::vector<torch::Tensor> receiveParameters(boost::asio::ip::tcp::socket& socket)
	{
		boost::asio::streambuf buffer;
		boost::asio::read(socket, buffer, boost::asio::transfer_all());
		std::stringstream ss;
		std::vector<torch::Tensor> parameters;
		ss << &buffer;
		torch::load(parameters, ss);
		
		return parameters;
	}
	
	std::vector<torch::Tensor> receiveBuffers(boost::asio::ip::tcp::socket& socket)
	{
		boost::asio::streambuf buffer;
		boost::asio::read(socket, buffer, boost::asio::transfer_all());
		std::stringstream ss;
		std::vector<torch::Tensor> buffers;
		ss << &buffer;
		torch::load(buffers, ss);
		
		return buffers;
	}
}

void SubmitParamsRequestHandler::handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server)
{
    auto space_pos = request.find(' ');
    const std::string identifier = request.substr(0, space_pos);
	space_pos = request.find(' ', space_pos+1);
	const std::string model_name = request.substr(space_pos + 1);
	
    // Check client record
    if (server.is_model_registered(model_name, identifier))
	{
        // Receive parameters and buffers
        std::vector<torch::Tensor> received_parameters = receiveParameters(socket);
		boost::asio::write(socket, boost::asio::buffer("RECEIVED\n"));
		std::vector<torch::Tensor> received_buffers = receiveBuffers(socket);
		
        // Update server model
        if (server.is_model_loaded(model_name))
		{
            torch::jit::script::Module& model = server.get_model_instance(model_name);
            auto& model_params = model.parameters();
			std::size_t param_count = 0;
			
			for(auto& param : model_params) param = received_parameters[param_count++];
			
            // Save updated model
            // torch::save(model, server.loaded_models_dir_ + model_name + "_updated.pt");
            // torch::save(model, server.loaded_models_dir_ + model_name + "_updated.pt");
			// Send success response
			boost::asio::write(socket, boost::asio::buffer("SUCCESS\n"));
		}
		else boost::asio::write(socket, boost::asio::buffer("ERROR Model not loaded\n"));
    }
	else boost::asio::write(socket, boost::asio::buffer("ERROR Client has not received this model\n"));
}
