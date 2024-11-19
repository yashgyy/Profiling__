#pragma once
#include "RequestHandler.h"
#include <string>

class CloseRequestHandler : public RequestHandler
{
	public:
    
	CloseRequestHandler() = default;
    ~CloseRequestHandler() = default;

    void handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server) override;
};
