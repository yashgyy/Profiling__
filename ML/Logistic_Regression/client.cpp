
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;
using boost::asio::ip::tcp;

// Sigmoid function for logistic regression
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// Load local data for logistic regression
MatrixXd load_local_data() {
    MatrixXd data(4, 2); // 4 samples, 2 features
    data << 1, 2,
            2, 1,
            3, 4,
            4, 3;

    // Shuffle rows of data to simulate different client data
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> indices = {0, 1, 2, 3};
    std::shuffle(indices.begin(), indices.end(), g);

    MatrixXd shuffled_data(4, 2);
    for (int i = 0; i < 4; ++i) {
        shuffled_data.row(i) = data.row(indices[i]);
    }

    return shuffled_data;
}

// Load corresponding labels for the data
VectorXd load_local_labels() {
    VectorXd labels(4); // 4 samples
    labels << 1, 1, 0, 0; // Binary labels for logistic regression
    return labels;
}

// Perform one step of gradient descent for logistic regression
VectorXd train_local_logistic_regression(const MatrixXd& data, const VectorXd& labels, VectorXd& weights, double learning_rate) {
    int n_samples = data.rows();
    int n_features = data.cols();

    VectorXd gradient = VectorXd::Zero(n_features);

    for (int i = 0; i < n_samples; ++i) {
        VectorXd xi = data.row(i);
        double yi = labels(i);
        double prediction = sigmoid(xi.dot(weights)); // Logistic prediction
        // Gradient of the log-loss
        gradient += xi * (prediction - yi);
    }

    // Update weights
    weights -= learning_rate * gradient / n_samples;

    return weights;
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        // Load local data and labels
        MatrixXd local_data = load_local_data();
        VectorXd local_labels = load_local_labels();

        // Initialize weights
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);
        VectorXd weights(local_data.cols());
        for (int i = 0; i < weights.size(); ++i) {
            weights(i) = dis(gen);
        }
       // VectorXd weights = VectorXd::Zero(local_data.cols());

        double learning_rate = 0.01;

        // Train the local model
        VectorXd updated_weights = train_local_logistic_regression(local_data, local_labels, weights, learning_rate);

        // Send the size of the weight vector first
        size_t vector_size = updated_weights.size();
        boost::asio::write(socket, boost::asio::buffer(&vector_size, sizeof(vector_size)));
        std::cout << "[DEBUG] Sent local weights size: " << vector_size << std::endl;

        // Send the actual weight vector
        boost::asio::write(socket, boost::asio::buffer(updated_weights.data(), updated_weights.size() * sizeof(double)));
        std::cout << "[DEBUG] Sent local model to server." << std::endl;

        // Ensure the size of the global weights matches
        VectorXd global_weights = VectorXd::Zero(weights.size());

        // Read the updated global model from the server
        boost::asio::read(socket, boost::asio::buffer(global_weights.data(), global_weights.size() * sizeof(double)));
        std::cout << "[DEBUG] Received updated global model from server, size: " << global_weights.size() << std::endl;

        // Print the coefficients (weights)
        std::cout << "[INFO] Final model coefficients (weights): " << global_weights.transpose() << std::endl;

    } catch (std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
    }

    return 0;
}
