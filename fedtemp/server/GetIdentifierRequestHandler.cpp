#include "GetIdentifierRequestHandler.h"
#include <string>
#include <random>

void GetIdentifierRequestHandler::handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server)
{
    // Generate unique identifier
    std::string identifier = std::to_string(std::rand());

    // Send identifier to client
    boost::asio::write(socket, boost::asio::buffer(identifier + "\n"));

    // Store client record
	server.register_client(identifier);
    // server.client_records_[identifier] = {};
}
