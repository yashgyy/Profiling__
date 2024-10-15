#include <iostream>
#include <vector>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <random>

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

void split_data(const MatrixXd& data, const VectorXd& labels, int feature_index, double threshold,
                MatrixXd& left_data, VectorXd& left_labels,
                MatrixXd& right_data, VectorXd& right_labels) {
    std::vector<int> left_indices, right_indices;

    for (int i = 0; i < data.rows(); ++i) {
        if (data(i, feature_index) <= threshold) {
            left_indices.push_back(i);
        } else {
            right_indices.push_back(i);
        }
    }

    // Create left and right data and labels
    left_data.resize(left_indices.size(), data.cols());
    left_labels.resize(left_indices.size());
    right_data.resize(right_indices.size(), data.cols());
    right_labels.resize(right_indices.size());

    for (size_t i = 0; i < left_indices.size(); ++i) {
        left_data.row(i) = data.row(left_indices[i]);
        left_labels(i) = labels(left_indices[i]);
    }
    for (size_t i = 0; i < right_indices.size(); ++i) {
        right_data.row(i) = data.row(right_indices[i]);
        right_labels(i) = labels(right_indices[i]);
    }
}


// Gini Impurity Calculation
double gini_impurity(const Eigen::VectorXd& labels) {
    int n_samples = labels.size();
    if (n_samples == 0) return 0.0;

    double count_pos = (labels.array() == 1.0).count();
    double count_neg = n_samples - count_pos;

    double prob_pos = count_pos / n_samples;
    double prob_neg = count_neg / n_samples;

    return 1.0 - (prob_pos * prob_pos + prob_neg * prob_neg);
}

// Train Decision Tree
TreeNode* train_tree(const MatrixXd& data, const VectorXd& labels, int depth = 0, int max_depth = 10) {
    TreeNode* node = new TreeNode();

    // Stopping conditions
    if (depth >= max_depth || labels.size() <= 1 || gini_impurity(labels) == 0) {
        node->prediction = (labels.sum() >= 0) ? 1.0 : -1.0; // Majority vote
        return node;
    }

    // Find the best split
    double best_gini = std::numeric_limits<double>::max();
    int best_feature = -1;
    double best_threshold = 0.0;
    MatrixXd best_left_data, best_right_data;
    VectorXd best_left_labels, best_right_labels;

    for (int feature_index = 0; feature_index < data.cols(); ++feature_index) {
        for (int i = 0; i < data.rows(); ++i) {
            double threshold = data(i, feature_index);

            MatrixXd left_data, right_data;
            VectorXd left_labels, right_labels;
            split_data(data, labels, feature_index, threshold, left_data, left_labels, right_data, right_labels);

            if (left_labels.size() == 0 || right_labels.size() == 0) continue;  // Skip invalid splits

            // Calculate Gini impurity for the split
            double left_gini = gini_impurity(left_labels);
            double right_gini = gini_impurity(right_labels);
            double weighted_gini = (left_labels.size() * left_gini + right_labels.size() * right_gini) / labels.size();

            if (weighted_gini < best_gini) {
                best_gini = weighted_gini;
                best_feature = feature_index;
                best_threshold = threshold;
                best_left_data = left_data;
                best_right_data = right_data;
                best_left_labels = left_labels;
                best_right_labels = right_labels;
            }
        }
    }

    // If no valid split was found, return a leaf node
    if (best_feature == -1) {
        node->prediction = (labels.sum() >= 0) ? 1.0 : -1.0;
        return node;
    }

    // Create a decision node
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = train_tree(best_left_data, best_left_labels, depth + 1, max_depth);
    node->right = train_tree(best_right_data, best_right_labels, depth + 1, max_depth);

    return node;
}

// Serialize a decision tree node
void serialize_tree_node(const TreeNode* node, std::vector<double>& serialized_tree) {
    if (!node) return;

    serialized_tree.push_back(node->feature_index);
    serialized_tree.push_back(node->threshold);
    serialized_tree.push_back(node->prediction);

    if (node->feature_index != -1) {  // Not a leaf
        serialize_tree_node(node->left, serialized_tree);
        serialize_tree_node(node->right, serialized_tree);
    }
}

// Random Forest class
class RandomForest {
public:
    std::vector<TreeNode*> trees;

    RandomForest(int num_trees) {
        trees.resize(num_trees, nullptr);
    }

    // Train the Random Forest with bootstrap sampling
    void train(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels) {
        std::mt19937 rng;
        std::uniform_int_distribution<int> dist(0, data.rows() - 1);

        for (int t = 0; t < trees.size(); ++t) {
            MatrixXd bootstrapped_data(data.rows(), data.cols());
            VectorXd bootstrapped_labels(data.rows());

            for (int i = 0; i < data.rows(); ++i) {
                int index = dist(rng);
                bootstrapped_data.row(i) = data.row(index);
                bootstrapped_labels(i) = labels(index);
            }

            trees[t] = train_tree(bootstrapped_data, bootstrapped_labels);
        }
    }

    // Serialize the entire forest
    std::vector<std::vector<double>> serialize_forest() {
        std::vector<std::vector<double>> serialized_forest(trees.size());

        for (int i = 0; i < trees.size(); ++i) {
            serialize_tree_node(trees[i], serialized_forest[i]);
        }

        return serialized_forest;
    }
};

// Function to load local data
MatrixXd load_local_data() {
    MatrixXd data(4, 2); // 4 samples, 2 features
    data << 1, 2,
            2, 1,
            3, 4,
            4, 3;
    return data;
}

VectorXd load_local_labels() {
    VectorXd labels(4); // 4 samples
    labels << 1, 1, -1, -1;
    return labels;
}

// Main client function
int main() {
    boost::asio::io_service io_service;
    tcp::socket socket(io_service);
    tcp::resolver resolver(io_service);
    boost::asio::connect(socket, resolver.resolve({"127.0.0.1", "8080"}));

    // Load local data and labels
    MatrixXd data = load_local_data();
    VectorXd labels = load_local_labels();

    // Train a random forest
    RandomForest forest(5); // A forest with 5 trees
    forest.train(data, labels);

    // Serialize the random forest model
    std::vector<std::vector<double>> serialized_forest = forest.serialize_forest();

    // Send the model size first (number of trees)
    int forest_size = serialized_forest.size();
    boost::asio::write(socket, boost::asio::buffer(&forest_size, sizeof(int)));

    // Send each serialized tree
    for (const auto& tree : serialized_forest) {
        int tree_size = tree.size();
        boost::asio::write(socket, boost::asio::buffer(&tree_size, sizeof(int)));
        boost::asio::write(socket, boost::asio::buffer(tree.data(), tree_size * sizeof(double)));
    }

    // Print confirmation message
    std::cout << "Decision trees have been successfully sent to the server." << std::endl;

    return 0;
}
