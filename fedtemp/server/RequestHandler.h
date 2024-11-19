#pragma once
#include <boost/asio.hpp>
#include <torch/torch.h>
#include <string>

class Server;

class RequestHandler
{
	public:
    
	virtual void handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server) = 0;
};
