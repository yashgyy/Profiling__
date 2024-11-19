#include "CloseRequestHandler.h"

void CloseRequestHandler::handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server)
{
    auto space_pos = request.find(' ');
    const std::string identifier = request.substr(0, space_pos);
	space_pos = request.find(' ', space_pos+1);
	const std::string model_name = request.substr(space_pos + 1);
	
    // Check client record
    if (server.is_model_registered(model_name, identifier))
	{
        // Remove model from client record
		server.remove_model_for_client(model_name, identifier);
        // server.client_records_[identifier].erase(model_name);
		server.save_client_records();

        boost::asio::write(socket, boost::asio::buffer("SUCCESS\n"));
    }
	else if (model_name == "all")
	{
        server.remove_client(identifier);
		server.save_client_records();
		
        boost::asio::write(socket, boost::asio::buffer("SUCCESS\n"));
    }
	else boost::asio::write(socket, boost::asio::buffer("ERROR Client has not received this model\n"));
}
