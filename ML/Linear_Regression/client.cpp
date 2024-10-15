#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using boost::asio::ip::tcp;

// Load local data for linear regression
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
    labels << 1, 1, 3, 3;
    return labels;
}

// Perform one step of gradient descent for linear regression
VectorXd train_local_linear_regression(const MatrixXd& data, const VectorXd& labels, VectorXd& weights, double learning_rate) {
    int n_samples = data.rows();
    int n_features = data.cols();

    VectorXd gradient = VectorXd::Zero(n_features);

    for (int i = 0; i < n_samples; ++i) {
        VectorXd xi = data.row(i);
        double yi = labels(i);
        // Gradient of the squared loss
        gradient += -2 * xi * (yi - xi.dot(weights));
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

        // Debug: print loaded data and labels
        std::cout << "[DEBUG] Loaded local data: \n" << local_data << std::endl;
        std::cout << "[DEBUG] Loaded local labels: \n" << local_labels.transpose() << std::endl;

        // Initialize local model weights with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);
        VectorXd weights(local_data.cols());
        for (int i = 0; i < weights.size(); ++i) {
            weights(i) = dis(gen);
        }

        // Debug: print initialized weights
        std::cout << "[DEBUG] Initialized weights: " << weights.transpose() << std::endl;

        // Train local model
        double learning_rate = 0.01;
        weights = train_local_linear_regression(local_data, local_labels, weights, learning_rate);

        // Debug: print weights after training
        std::cout << "[DEBUG] Weights after training: " << weights.transpose() << std::endl;

        //  local update to the server
        //std:Send:cout << "[DEBUG] Sending local weights to server: " << weights.transpose() << std::endl;
        
        // Send the number of features first
        int num_features = weights.size();
        boost::asio::write(socket, boost::asio::buffer(&num_features, sizeof(int)));

        // Then send the local update (weights)
        boost::asio::write(socket, boost::asio::buffer(weights.data(), weights.size() * sizeof(double)));
        std::cout << "[DEBUG] Sending local weights to server: " << weights.transpose() << std::endl;


        // Receive updated global model from the server
        boost::asio::read(socket, boost::asio::buffer(weights.data(), weights.size() * sizeof(double)));

        // Debug: print updated global weights received
        std::cout << "[DEBUG] Updated global weights received from server: " << weights.transpose() << std::endl;

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
