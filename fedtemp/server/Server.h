#pragma once
#include <boost/asio.hpp>
#include <torch/torch.h>
#include <map>
#include <string>
#include <memory>
#include <fstream>

class ClientSession;
class RequestDispatcher;
class RequestHandler;

class Server
{
	public:
    
	Server(const std::string& ip, const std::string& port);
    void start();
	
	private:
    
	boost::asio::io_context io_context_;
    boost::asio::ip::tcp::acceptor acceptor_;
    std::map<std::string, torch::jit::script::Module> loaded_models_;
    std::map<std::string, std::map<std::string, bool>> client_records_;
    std::string client_records_file_ = "client_records.txt";
    std::string loaded_models_dir_ = "loaded_models/";
};
