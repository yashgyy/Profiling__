#include "FedComm.h"

FedComm::FedComm(const std::string& server_ip, const std::string& port) : socket_(io_context_) 
{
    // Connect to the server
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(server_ip), std::stoi(port));
    socket_.connect(endpoint);
	
    // Request unique identifier
    std::string request = "GET_IDENTIFIER";
    boost::asio::write(socket_, boost::asio::buffer(request));
	
    // Receive unique identifier
    boost::asio::streambuf buffer;
    boost::asio::read_until(socket_, buffer, '\n');
    std::istream input(&buffer);
    input >> unique_identifier_;
	
    if (unique_identifier_.empty()) throw std::runtime_error("Failed to receive unique identifier");
}

FedComm::~FedComm() { socket_.close(); }

torch::jit::script::Module FedComm::get_model(const std::string& model_name)
{
    std::string request = "GET_MODEL " + model_name;
    boost::asio::write(socket_, boost::asio::buffer(request));
	
    // Receive model file
    boost::asio::streambuf buffer;
    boost::asio::read_until(socket_, buffer, '\n');
    std::istream input(&buffer);
    std::string response;
    input >> response;

    if (response == "SUCCESS")
	{
        boost::asio::read(socket_, buffer, boost::asio::transfer_all());
        std::stringstream file;
		file << &buffer;
        
        return torch::jit::load(file);
    }
	
	throw std::runtime_error(response);
}

void FedComm::submit_params(const std::string& model_name, torch::jit::script::Module& model)
{
    std::string request = "SUBMIT_PARAMS " + model_name;
    boost::asio::write(socket_, boost::asio::buffer(request));
	
    // Serialize and send parameters and buffers
    std::vector<torch::Tensor> parameters, buffers;
	
	for(const auto& parameter : model.parameters()) parameters.push_back(parameter);
	for(const auto& buffer : model.buffers()) buffers.push_back(buffer);
	
    std::stringstream ss;
	torch::save(parameters, ss);
    // torch::serialize::OutputArchive output(ss);
    // output << named_parameters << named_buffers;
    boost::asio::write(socket_, boost::asio::buffer(ss.str()));
	
    // Receive response
	std::string response;
	{
		boost::asio::streambuf buffer;
		boost::asio::read_until(socket_, buffer, '\n');
		std::istream input(&buffer);
		input >> response;
	}
	
    if (response != "RECEIVED") throw std::runtime_error(response);
	
	ss = std::stringstream();
	torch::save(buffers, ss);
	boost::asio::write(socket_, boost::asio::buffer(ss.str()));
	
    // Receive response
	{
		boost::asio::streambuf buffer;
		boost::asio::read_until(socket_, buffer, '\n');
		std::istream input(&buffer);
		input >> response;
	}
	
    if (response != "SUCCESS") throw std::runtime_error(response);
}

std::vector<torch::Tensor> FedComm::get_params(const std::string& model_name)
{
    std::string request = "GET_PARAMS " + model_name;
    boost::asio::write(socket_, boost::asio::buffer(request));

    // Receive named parameters
	std::string response;
	{
		boost::asio::streambuf buffer;
		boost::asio::read_until(socket_, buffer, '\n');
		std::istream input(&buffer);
		input >> response;
	}
	
    if (response == "SUCCESS")
	{
		boost::asio::streambuf buffer;
		boost::asio::read(socket_, buffer, boost::asio::transfer_all());
		std::stringstream ss;
		ss << &buffer;
		std::vector<torch::Tensor> parameters;
		
		torch::load(parameters, ss);
		return parameters;
    }
	
	throw std::runtime_error(response);
}

void FedComm::close(const std::string& model_name)
{
    std::string request = "CLOSE " + model_name;
    boost::asio::write(socket_, boost::asio::buffer(request));

    // Receive response
    boost::asio::streambuf buffer;
    boost::asio::read_until(socket_, buffer, '\n');
    std::istream input(&buffer);
    std::string response;
    input >> response;

    if (response == "SUCCESS")
	{
        if (model_name == "all") unique_identifier_.clear();
		return;
    }
	
	throw std::runtime_error(response);
}
