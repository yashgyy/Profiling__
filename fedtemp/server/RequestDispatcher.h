#pragma once
#include <torch/torch.h>
#include <map>
#include <string>
#include "RequestHandler.h"

class Server;

class RequestDispatcher
{
	public:
    
	RequestDispatcher();
    void handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server);

	private:
    
	Server& server_;
    std::map<std::string, std::unique_ptr<RequestHandler>> handlers_;
};
