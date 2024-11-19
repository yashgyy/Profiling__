#include "Server.h"
#include "ClientSession.h"

Server::Server(const std::string& ip, const std::string& port) : acceptor_(
	io_context_, boost::asio::ip::tcp::endpoint(boost::asio::ip::address::from_string(ip), std::stoi(port))
) {
    // Load client records from file
    std::ifstream file(client_records_file_);
    if (file.is_open())
	{
        std::string line;
        while (std::getline(file, line))
		{
            std::istringstream iss(line);
            std::string client_id, model_name;
            iss >> client_id >> model_name;
            client_records_[client_id][model_name] = true;
        }
        file.close();
    }
}

void Server::start()
{
    accept_client();
    io_context_.run();
}

void Server::accept_client()
{
    acceptor_.async_accept(
        [this](boost::asio::ip::tcp::socket socket, const boost::system::error_code& error)
		{
            if (!error) std::make_shared<ClientSession>(std::move(socket), *this)->start();
			// Continue accepting clients
            accept_client();
        }
	);
}

int main()
{
	constexpr char default_server_ip[] = "127.0.0.1", default_port[] = "2413";
	std::string server_ip, port;
	
	std::cout << "Enter server ip (default = " << default_server_ip << "): "; 
	std::cin >> server_ip;
	if (server_ip.empty()) server_ip =  default_server_ip;
	
	std::cout << "Enter server port: "; 
	std::cin >> port;
	if (port.empty()) port =  default_port;
	
	Server server {server_ip, port};
	server.start();
	
	return 0;
}
