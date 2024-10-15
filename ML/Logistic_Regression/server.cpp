
#include <iostream>
#include <vector>
#include <boost/asio.hpp>
#include <thread>
#include <Eigen/Dense>
#include <chrono> // For sleep and delay

using namespace Eigen;
using boost::asio::ip::tcp;

VectorXd global_weights;
VectorXd total_weights;
int client_count = 0;

// Aggregates the updates from clients
void aggregate_model(const VectorXd& local_update) {
    total_weights += local_update;
    client_count++;
}

// Handles the communication with each client
void handle_client(tcp::socket socket) {
    try {
        // Receive the size of the incoming weight vector from the client
        size_t vector_size = 0;
        boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(vector_size)));
        std::cout << "[DEBUG] Expected local update size from client: " << vector_size << std::endl;

        // Initialize the local_update vector to the correct size
        VectorXd local_update(vector_size);

        // Read the actual local model update from the client
        if (vector_size > 0) {
            boost::asio::read(socket, boost::asio::buffer(local_update.data(), vector_size * sizeof(double)));
            std::cout << "[DEBUG] Received local update from client, size: " << local_update.size() << std::endl;
        } else {
            std::cerr << "[ERROR] Received local update with size 0!" << std::endl;
            return;
        }

        // Initialize global weights if it's the first client
        if (client_count == 0 && local_update.size() > 0) {
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
        std::cout << "[DEBUG] Aggregated global weights, size: " << global_weights.size() << std::endl;

        // Introduce a short delay to ensure client is ready to receive the updated model
       // std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Send updated global model back to the client
        boost::asio::write(socket, boost::asio::buffer(global_weights.data(), global_weights.size() * sizeof(double)));
        std::cout << "[DEBUG] Sent updated global model to client, size: " << global_weights.size() << std::endl;

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
