#pragma once
#include <boost/asio.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <fstream>
#include <stdexcept>

class FedComm
{
	public:
    
	FedComm(const std::string& server_ip, const std::string& port);
    ~FedComm();
	
    torch::jit::script::Module get_model(const std::string& model_name);
    void submit_params(const std::string& model_name, torch::jit::script::Module& model);
    std::vector<torch::Tensor> get_params(const std::string& model_name);
    void close(const std::string& model_name);

	private:
    
	boost::asio::io_context io_context_;
    boost::asio::ip::tcp::socket socket_;
    std::string unique_identifier_;
};
