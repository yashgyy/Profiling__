#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <mutex>
#include <Eigen/Dense>
#include <map>
#include <iterator>  // For begin() and end()

using namespace Eigen;
using boost::asio::ip::tcp;

// Decision Tree Node Structure
struct TreeNode {
    int feature_index = -1;
    double threshold = 0.0;
    double prediction = 0.0;
    TreeNode* left = nullptr;
    TreeNode* right = nullptr;
};

std::mutex model_mutex;

// Aggregated decision trees (from multiple clients)
std::vector<std::vector<TreeNode*>> aggregated_forests;  // Correct declaration

// Deserialize TreeNode (assuming a specific format for simplicity)
TreeNode* deserialize_tree_node(const std::vector<double>& serialized_tree, int& index) {
    if (index >= serialized_tree.size()) return nullptr;

    TreeNode* node = new TreeNode;
    node->feature_index = static_cast<int>(serialized_tree[index++]);
    node->threshold = serialized_tree[index++];
    node->prediction = serialized_tree[index++];

    // Recursively deserialize left and right subtrees
    if (node->feature_index != -1) {
        node->left = deserialize_tree_node(serialized_tree, index);
        node->right = deserialize_tree_node(serialized_tree, index);
    }

    return node;
}

// Predict using a single decision tree
double predict_tree(const TreeNode* node, const Eigen::VectorXd& x) {
    if (!node->left && !node->right) {
        return node->prediction;
    }
    if (x(node->feature_index) <= node->threshold) {
        return predict_tree(node->left, x);
    } else {
        return predict_tree(node->right, x);
    }
}

// Aggregation function using majority voting
std::vector<double> majority_voting(const MatrixXd& test_data) {
    std::vector<double> final_predictions(test_data.rows(), 0.0);

    // For each test sample, calculate votes from all trees
    for (int i = 0; i < test_data.rows(); ++i) {
        std::map<double, int> votes_count;  // Map to store class and its vote count

        for (const auto& forest : aggregated_forests) {
            for (const auto& tree : forest) {
                double prediction = predict_tree(tree, test_data.row(i));
                votes_count[prediction]++;
            }
        }

        // Determine the class with the most votes
        int max_votes = 0;
        double best_class = -1;
        for (const auto& [predicted_class, count] : votes_count) {
            if (count > max_votes) {
                max_votes = count;
                best_class = predicted_class;
            }
        }

        // Final prediction for this sample
        final_predictions[i] = best_class;
    }

    return final_predictions;
}

// Handle client function: Deserialize and store the client's random forest
void handle_client(tcp::socket socket) {
    try {
        std::cout << "[DEBUG] Handling new client connection.\n";

        // Read size of the incoming forest (number of trees)
        int forest_size;
        boost::asio::read(socket, boost::asio::buffer(&forest_size, sizeof(int)));

        // Deserialize the forest (each tree represented as a serialized vector of doubles)
        std::vector<TreeNode*> local_forest(forest_size);
        for (int i = 0; i < forest_size; ++i) {
            // Read serialized tree size
            int tree_size;
            boost::asio::read(socket, boost::asio::buffer(&tree_size, sizeof(int)));

            // Read the serialized tree data
            std::vector<double> serialized_tree(tree_size);
            boost::asio::read(socket, boost::asio::buffer(serialized_tree.data(), tree_size * sizeof(double)));

            // Deserialize the tree and store it in local_forest
            int index = 0;
            local_forest[i] = deserialize_tree_node(serialized_tree, index);
        }

        // Lock and aggregate the received forest into the global model
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            aggregated_forests.push_back(local_forest);  // Correct usage of push_back
        }

        std::cout << "[DEBUG] Received and stored a forest from a client.\n";

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

// Function to perform predictions
void perform_predictions() {
    if (aggregated_forests.size() >= 2) {  // Wait until we have at least 2 forests
        MatrixXd test_data(4, 2);  // Example test data
        test_data << 1, 2,
                     2, 1,
                     3, 4,
                     4, 3;

        std::vector<double> predictions = majority_voting(test_data);

        // Display the predictions
        std::cout << "Predictions for test data: ";
        for (const auto& pred : predictions) {
            std::cout << pred << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    boost::asio::io_service io_service;
    tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), 8080));

    std::cout << "Server is running on port 8080...\n";

    while (true) {
        tcp::socket socket(io_service);
        acceptor.accept(socket);

        // Handle each client in a separate thread
        std::thread(handle_client, std::move(socket)).detach();

        // After handling clients, call the prediction function if conditions are met
        perform_predictions();
    }

    return 0;
}
