#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <tuple>
#include <stack>
#include <vector>

using namespace std;
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    int goodNodes(TreeNode* root) {

        // dfs
        // use stack
        // O(n)
        // good node if max(visited nodes)

        if (!root){
            return 0;
        }

        stack<pair<TreeNode*, int>> nodes; 
        nodes.push({root, root->val});
        int good_nodes = 0; 
        while (!nodes.empty()){
           
           auto [curr, maxval] = nodes.top();
           nodes.pop();

        if (curr->val>=maxval){
            ++good_nodes;
        }
           int newmax = max(maxval, curr->val);
           if (curr->right){
           nodes.push({curr->right, newmax});
           }
           if (curr->left){
           nodes.push({curr->left, newmax});
           }
    }
        return good_nodes;
    }
};


class SolutionRecursive{
    public:
        int goodNodes(TreeNode* root){
            if (!root){
                return 0;
            }
            int good_nodes=dfs(root, root->val);
            return good_nodes;
        }

    private:
        int dfs(TreeNode* node, int maxval){
            if (!node){
                return 0;
            }
            int result = node->val >= maxval ? 1 : 0;
            maxval = max(maxval, node->val);
            if (node->left){
            result += dfs(node->left,maxval);
            }
            if (node->right){
            result += dfs(node->right,maxval);
            }
            return result;
        }
};


// p=4
// c=4
// [3]


//      3
//     / \
//    1   4
//   /   / \
//  3   1   5
//     / \
//    1   4
//   /   / \
//  3   1   5

int main() {
    // Test Case 1: Example tree
    //       3
    //      / \
    //     1   4
    //    /   / \
    //   3   1   5
    TreeNode* root1 = new TreeNode(3,
        new TreeNode(1,
            new TreeNode(3),
            nullptr),
        new TreeNode(4,
            new TreeNode(1),
            new TreeNode(5)));
    
    std::cout << "Test Case 1:" << std::endl;
    Solution sol;
    std::cout << "Number of good nodes (iterative): " << sol.goodNodes(root1) << std::endl;
    SolutionRecursive solRec;
    std::cout << "Number of good nodes (recursive): " << solRec.goodNodes(root1) << std::endl;
    
    // Test Case 2: Empty tree
    TreeNode* root2 = nullptr;
    std::cout << "\nTest Case 2 - Empty tree:" << std::endl;
    std::cout << "Number of good nodes (iterative): " << sol.goodNodes(root2) << std::endl;
    std::cout << "Number of good nodes (recursive): " << solRec.goodNodes(root2) << std::endl;
    
    // Test Case 3: Single node
    TreeNode* root3 = new TreeNode(1);
    std::cout << "\nTest Case 3 - Single node:" << std::endl;
    std::cout << "Number of good nodes (iterative): " << sol.goodNodes(root3) << std::endl;
    std::cout << "Number of good nodes (recursive): " << solRec.goodNodes(root3) << std::endl;
    
    // Test Case 4: Complex tree
    // [-1,5,-2,4,4,2,-2,null,null,-4,null,-2,3,null,-2,0,null,-1,null,-3,null,-4,-3,3,null,null,null,null,null,null,null,3,-3]
    TreeNode* root4 = new TreeNode(-1,
        new TreeNode(5,
            new TreeNode(4,
                new TreeNode(4),
                new TreeNode(-4)),
            new TreeNode(2,
                new TreeNode(-2),
                nullptr)),
        new TreeNode(-2,
            new TreeNode(4,
                new TreeNode(-2),
                new TreeNode(3)),
            new TreeNode(-2,
                new TreeNode(0,
                    new TreeNode(-1),
                    new TreeNode(-3)),
                new TreeNode(-4,
                    new TreeNode(-3),
                    new TreeNode(3)))));
    
    std::cout << "\nTest Case 4 - Complex tree:" << std::endl;
    std::cout << "Number of good nodes (iterative): " << sol.goodNodes(root4) << std::endl;
    std::cout << "Number of good nodes (recursive): " << solRec.goodNodes(root4) << std::endl;
    
    // Cleanup
    delete root1->left->left;
    delete root1->left;
    delete root1->right->left;
    delete root1->right->right;
    delete root1->right;
    delete root1;
    delete root3;
    delete root4->left->left->left;
    delete root4->left->left->right;
    delete root4->left->left;
    delete root4->left->right->left;
    delete root4->left->right;
    delete root4->left;
    delete root4->right->left->left;
    delete root4->right->left->right;
    delete root4->right->left;
    delete root4->right->right->left->left;
    delete root4->right->right->left->right;
    delete root4->right->right->left;
    delete root4->right->right->right->left;
    delete root4->right->right->right->right;
    delete root4->right->right->right;
    delete root4->right->right;
    delete root4->right;
    delete root4;
    
    return 0;
} 