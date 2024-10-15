#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using boost::asio::ip::tcp;

// Radial Basis Function (RBF) Kernel
double rbf_kernel(const VectorXd& x1, const VectorXd& x2, double gamma = 0.1) {
    return std::exp(-gamma * (x1 - x2).squaredNorm());
}

// Shuffle data to simulate different local datasets for each client
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

VectorXd load_local_labels() {
    VectorXd labels(4); // 4 samples
    labels << 1, 1, -1, -1;
    return labels;
}

// Kernelized SVM training using SGD
VectorXd train_local_kernel_svm(const MatrixXd& data, const VectorXd& labels, VectorXd& weights, double learning_rate, double gamma) {
    int n_samples = data.rows();
    int n_features = data.cols();  // Not used here, but good to keep track

    VectorXd gradient = VectorXd::Zero(n_samples);  // Gradient should match the size of weights (which is n_samples)

    for (int i = 0; i < n_samples; ++i) {
        VectorXd xi = data.row(i);
        double yi = labels(i);

        // Kernel function computation: prediction based on weighted kernel
        double kernel_output = 0.0;
        for (int j = 0; j < n_samples; ++j) {
            kernel_output += rbf_kernel(xi, data.row(j), gamma) * weights(j);  // Use weights
        }

        // Hinge loss calculation: only apply gradient if hinge loss is active (i.e., yi * prediction < 1)
        if (yi * kernel_output < 1) {
            // Update the gradient for this sample
            for (int j = 0; j < n_samples; ++j) {
                gradient(j) += -yi * rbf_kernel(data.row(j), xi, gamma);  // Gradients w.r.t weights
            }
        }
    }

    // Update weights with the calculated gradient
    weights -= learning_rate * gradient;

    return weights;
}


int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        // Load local data
        MatrixXd local_data = load_local_data();
        VectorXd local_labels = load_local_labels();

        // Initialize local model weights with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01); // Small random values
        VectorXd local_weights = VectorXd::Zero(local_data.rows()).unaryExpr([&](double dummy) { return d(gen); });

        double learning_rate = 0.01;
        double gamma = 0.1; // RBF kernel parameter

        // Perform local training (single step)
        VectorXd updated_weights = train_local_kernel_svm(local_data, local_labels, local_weights, learning_rate, gamma);

        // Print local update
        std::cout << "Local model update: " << updated_weights.transpose() << std::endl;

        // Send the size of the vector first
        int vector_size = updated_weights.size();
        boost::asio::write(socket, boost::asio::buffer(&vector_size, sizeof(int)));

        // Send local update (weights) to server
        boost::asio::write(socket, boost::asio::buffer(updated_weights.data(), updated_weights.size() * sizeof(double)));

        // Receive updated global model from server
        VectorXd global_model(local_weights.size());
        boost::asio::read(socket, boost::asio::buffer(global_model.data(), global_model.size() * sizeof(double)));

        std::cout << "Received updated global model: " << global_model.transpose() << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
