#include "GetModelRequestHandler.h"

void GetModelRequestHandler::handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server)
{
    auto space_pos = request.find(' ');
    std::string identifier = request.substr(0, space_pos);
	space_pos = request.find(' ', space_pos+1);
	std::string model_name = request.substr(space_pos + 1);
	
    // if (std::filesystem::exists(server.loaded_models_dir_ + model_name + ".pt"))
    if (server.exists_model(model_name))
	{
        // if (!server.loaded_models_.count(model_name))
        if (!server.is_model_loaded(model_name))
		{
			server.load_model(model_name);
            // server.loaded_models_[model_name] = torch::jit::load(server.loaded_models_dir_ + model_name + ".pt");
        }
		
        // Send model file to client
        std::ifstream file(server.loaded_models_dir_ + model_name + ".pt", std::ios::binary);
        boost::asio::write(socket, boost::asio::buffer("SUCCESS\n"));
        boost::asio::write(socket, boost::asio::buffer(file, file.seekg(0, file.end).tellg()));
        file.close();
		
		// Update client record
        server.register_model_to_client(model_name, identifier);
        // server.client_records_[identifier][model_name] = true;
    }
	else boost::asio::write(socket, boost::asio::buffer("ERROR Model not found.\n"));
}
