#pragma once
#include "RequestHandler.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>

class SubmitParamsRequestHandler : public RequestHandler
{
	public:
    
	SubmitParamsRequestHandler() = default;
    ~SubmitParamsRequestHandler() = default;
	
	void handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server) override;
};
