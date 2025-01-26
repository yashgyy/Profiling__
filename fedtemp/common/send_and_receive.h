#include <boost/asio.hpp>
#include <string>    

std::string send_data(boost::asio::ip::tcp::socket& socket, const std::string& data, bool acknowledgement = true);

std::string receive_data(boost::asio::ip::tcp::socket& socket);
