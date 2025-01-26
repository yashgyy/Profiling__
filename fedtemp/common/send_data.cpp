#include <cstdint>
#include "send_and_receive.h"
#include <string>

std::string send_data(boost::asio::ip::tcp::socket& socket, const std::string& data) {
    uint32_t data_size = htonl((uint32_t)data.size());
    std::string message {reinterpret_cast<char*>(&data_size), sizeof(data_size)};
    message += data;
    
    std::size_t bytes_transferred = boost::asio::write(socket, boost::asio::buffer(message));
    if (bytes_transferred != message.length()) return "Error in sending message.";
    
    return "";
}
