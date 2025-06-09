#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <tuple>
#include <queue>
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

// TODO: Implement your solution here

class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
    // bfs
    // level order
    // queue - to hold all the nodes
    // array - to hold all the nodes of levels - can be a temp
    vector<int> rightNodes;
    queue<TreeNode*> q;
    if (!root) {
        return rightNodes;
    }
    q.push(root);
    while (!q.empty()){
        vector<int> level;
        for (int i=0;i<q.size();i++){
            TreeNode*  curr = q.front();
            q.pop(); // pop current node
            if (curr->left!=NULL){
            q.push(curr->left);
            }
            if (curr->right!=NULL){
            q.push(curr->right);
            }
                        level.push_back(curr->val); // do we need this?
        }
        rightNodes.push_back(level[0]);  // for each level
    }
    return rightNodes; 
    }

};



void printRightSideView(const std::vector<int>& result) {
    std::cout << "[ ";
    for (int val : result) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    // Test Case 1: Example tree
    //       1
    //      / \
    //     2   3
    //      \   \
    //       5   4
    TreeNode* root1 = new TreeNode(1,
        new TreeNode(2,
            nullptr,
            new TreeNode(5)),
        new TreeNode(3,
            nullptr,
            new TreeNode(4)));
    Solution sol;

    std::cout << "Test Case 1:" << std::endl;
    printRightSideView(sol.rightSideView(root1));
    
    // Test Case 2: Empty tree
    TreeNode* root2 = nullptr;
    std::cout << "\nTest Case 2 - Empty tree:" << std::endl;
    printRightSideView(sol.rightSideView(root2));
    
    // Test Case 3: Single node
    TreeNode* root3 = new TreeNode(1);
    std::cout << "\nTest Case 3 - Single node:" << std::endl;
    printRightSideView(sol.rightSideView(root3));
    
    // Cleanup
    delete root1->left->right;
    delete root1->left;
    delete root1->right->right;
    delete root1->right;
    delete root1;
    delete root3;
    
    return 0;
} 