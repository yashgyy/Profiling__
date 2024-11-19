#pragma once
#include "RequestHandler.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>

class GetParamsRequestHandler : public RequestHandler
{
	public:
    
	GetParamsRequestHandler() = default;
    ~GetParamsRequestHandler() = default;

    void handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server) override;
};
