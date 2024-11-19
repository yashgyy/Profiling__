#include "GetParamsRequestHandler.h"

void GetParamsRequestHandler::handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server)
{
    auto space_pos = request.find(' ');
    const std::string identifier = request.substr(0, space_pos);
	space_pos = request.find(' ', space_pos+1);
	const std::string model_name = request.substr(space_pos + 1);
	
    // Check client record
    // if (server.client_records_[identifier].count(model_name))
    if (server.is_model_registered(model_name, identifier))
	{
        // Send named parameters
        if (server.is_model_loaded(model_name))
		{
            torch::jit::script::Module& model = server.get_model_instance(model_name);
            // torch::jit::script::Module& model = server.loaded_models_[model_name];
			
			boost::asio::write(socket, boost::asio::buffer("SUCCESS\n"));
			
            // Send parameters
            std::vector<torch::Tensor> parameters;
			for(const auto& parameter : model.parameters()) parameters.push_back(parameter);
			std::stringstream ss;
			torch::save(parameters, ss);
			boost::asio::write(socket, boost::asio::buffer(ss.str()));
        }
		else
		{
            boost::asio::write(socket, boost::asio::buffer("ERROR Model not loaded\n"));
        }
    }
	else boost::asio::write(socket, boost::asio::buffer("ERROR Client has not received this model.\n"));
}
