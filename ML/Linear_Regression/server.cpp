#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include <Eigen/Dense>

//g++ server_continue.cpp -o server  -I /usr/include/eigen3

using namespace Eigen;
using boost::asio::ip::tcp;

VectorXd global_weights;
VectorXd total_weights;
int client_count = 0;

// Aggregating the models from clients
void aggregate_model(const VectorXd& local_update) {
    total_weights += local_update;
    client_count++;
}

void handle_client(tcp::socket socket) {
    try {
        // First, read the number of features (size of the local update)
        int num_features;
        boost::asio::read(socket, boost::asio::buffer(&num_features, sizeof(int)));

        // Now allocate the buffer dynamically based on the received size
        std::vector<double> buffer(num_features);
        boost::asio::read(socket, boost::asio::buffer(buffer.data(), buffer.size() * sizeof(double)));

        // Convert the buffer to a VectorXd for Eigen
        VectorXd local_update = Eigen::Map<VectorXd>(buffer.data(), buffer.size());

        std::cout << "[DEBUG] Received local update: " << local_update.transpose() << std::endl;

        // Initialize global_weights and total_weights if not already initialized
        if (global_weights.size() == 0) {
            global_weights = VectorXd::Zero(local_update.size());  // Initialize to correct size
            total_weights = VectorXd::Zero(local_update.size());   // Initialize to the same size
            std::cout << "[DEBUG] Initialized global_weights and total_weights with size: " << global_weights.size() << std::endl;
        }

        // Ensure sizes match before aggregating
        if (global_weights.size() == local_update.size()) {
            aggregate_model(local_update);
        } else {
            std::cerr << "[ERROR] Size mismatch between global model and local update!" << std::endl;
        }

        // Update global weights by averaging the total weights
        global_weights = total_weights / client_count;
        std::cout << "[DEBUG] Aggregated global weights: " << global_weights.transpose() << std::endl;

        // Send updated global model back to the client
        boost::asio::write(socket, boost::asio::buffer(global_weights.data(), global_weights.size() * sizeof(double)));
        std::cout << "[DEBUG] Sent updated global model to client." << std::endl;

    } catch (std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}



int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        std::cout << "[DEBUG] Server started. Waiting for clients on port 8080..." << std::endl;

        while (true) {
            tcp::socket socket(io_context);
            std::cout << "[DEBUG] Waiting for new client connection..." << std::endl;
            acceptor.accept(socket);
            std::cout << "[DEBUG] Client connected." << std::endl;

            // Start a thread to handle each client
            std::thread(handle_client, std::move(socket)).detach();
        }
    } catch (std::exception& e) {
        std::cerr << "[ERROR] Exception in main: " << e.what() << std::endl;
    }

    return 0;
}
