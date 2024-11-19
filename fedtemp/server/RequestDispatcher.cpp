#include "RequestDispatcher.h"
#include "GetIdentifierRequestHandler.h"
#include "GetModelRequestHandler.h"
#include "SubmitParamsRequestHandler.h"
#include "GetParamsRequestHandler.h"
#include "CloseRequestHandler.h"

RequestDispatcher::RequestDispatcher()
{
    handlers_["GET_IDENTIFIER"] = std::make_unique<GetIdentifierRequestHandler>();
    handlers_["GET_MODEL"] = std::make_unique<GetModelRequestHandler>();
    handlers_["SUBMIT_PARAMS"] = std::make_unique<SubmitParamsRequestHandler>();
    handlers_["GET_PARAMS"] = std::make_unique<GetParamsRequestHandler>();
    handlers_["CLOSE"] = std::make_unique<CloseRequestHandler>();
}

void RequestDispatcher::handleRequest(const std::string& request, boost::asio::ip::tcp::socket& socket, Server& server)
{
    auto space_pos = request.find(' ');
    std::string first_token = request.substr(0, space_pos), request_type;
	
	if (first_token == "GET_IDENTIFIER") request_type = "GET_IDENTIFIER";
	else if (server.is_client_registered(first_token))
	{
		auto space_pos2 = request.find(' ', space_pos+1);
		request_type = request.substr(space_pos+1, space_pos2-space_pos-1);
	}
	else
	{
		boost::asio::write(socket, boost::asio::buffer("ERROR Invalid identifier: " + first_token + "\n"));
		return;
	}
	
    if (handlers_.count(request_type)) handlers_[request_type]->handleRequest(request, socket, server);
    else
	{
        boost::asio::write(socket, boost::asio::buffer("ERROR Unknown request type: " + request_type + "\n"));
    }
}
