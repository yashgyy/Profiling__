#pragma once
#include "RequestHandler.h"

class GetIdentifierRequestHandler : public RequestHandler
{
	public:
    
	void handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server) override;
};
