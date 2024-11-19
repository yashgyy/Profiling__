#pragma once
#include <boost/asio.hpp>
#include <torch/torch.h>
#include <memory>
#include "RequestDispatcher.h"

class Server;

class ClientSession : public std::enable_shared_from_this<ClientSession>
{
	public:
    
	ClientSession(boost::asio::ip::tcp::socket socket, Server& server);
    void start();

	private:
    
	boost::asio::ip::tcp::socket socket_;
	boost::asio::streambuf buffer_;
    Server& server_;
    RequestDispatcher dispatcher_;
};
