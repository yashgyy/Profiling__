#include "ClientSession.h"

ClientSession::ClientSession(boost::asio::ip::tcp::socket socket, Server& server) 
: socket_(std::move(socket)), server_(server), dispatcher_() {}

void ClientSession::start()
{
    // Receive request
    boost::asio::async_read_until(
        socket_, buffer_, '\n',
        [self = shared_from_this()](const boost::system::error_code& error, std::size_t size)
		{
            if (!error)
			{
                std::string request;
                std::istream input(&self->buffer_);
                input >> request;

                // Handle request
                self->dispatcher_.handleRequest(request, self->socket_, self->server_);
            }
        }
	);
}
