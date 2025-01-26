#include <cstdint>
#include "send_and_receive.h"
#include <string>

std::string receive_data(boost::asio::ip::tcp::socket& socket, std::string& data) {
    uint32_t data_size;
    std::size_t received = boost::asio::read(socket, boost::asio::buffer(&data_size, sizeof(data_size)));
    if (received != sizeof(data_size)) return "Error receiving data size.";
    data_size = ntohl(data_size); // Convert from network byte order
    
    data.resize((std::size_t)data_size);
    received = boost::asio::read(socket, boost::asio::buffer(data.data(), (std::size_t)data_size));
    if (received != (std::size_t)data_size) return "Error receiving data.";
    
    return "";
}
