#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <tuple>
#include <queue>
#include <vector>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
};

// TODO: Implement your solution here

void printLevelOrder(const std::vector<std::vector<int>>& levels) {
    for (const auto& level : levels) {
        std::cout << "[ ";
        for (int val : level) {
            std::cout << val << " ";
        }
        std::cout << "]" << std::endl;
    }
}


std::vector<std::vector<int>> levelOrder(TreeNode* node){
    // implementing bfs
    std::vector<std::vector<int>> levels;
    std::queue<TreeNode*> queue;
    // in case !node
    if (!node){
        return levels;
    }

    queue.push(node);

        while (!queue.empty()){
            std::vector<int> level;
            int size = queue.size(); 
                    
            for (int i=0;i<size;i++){
                TreeNode* curr = queue.front();
                queue.pop();
                if (curr){
                    level.push_back(curr->val);
                    queue.push(curr->left);
                    queue.push(curr->right);
            }
        }
        if(!level.empty()){
            levels.push_back(level);
        }
    }
    return levels;
} // O(n) , O(n) 


int main() {
    // Test Case 1: Simple tree
    //       3
    //      / \
    //     9  20
    //        / \
    //       15  7
    TreeNode* root1 = new TreeNode(3,
        new TreeNode(9),
        new TreeNode(20,
            new TreeNode(15),
            new TreeNode(7)));
    
    std::cout << "Test Case 1:" << std::endl;
    printLevelOrder(levelOrder(root1));
    
    // Test Case 2: Empty tree
    TreeNode* root2 = nullptr;
    std::cout << "\nTest Case 2 - Empty tree:" << std::endl;
    printLevelOrder(levelOrder(root2));
    
    // Test Case 3: Single node
    TreeNode* root3 = new TreeNode(1);
    std::cout << "\nTest Case 3 - Single node:" << std::endl;
    printLevelOrder(levelOrder(root3));
    
    // Cleanup
    delete root1->right->left;
    delete root1->right->right;
    delete root1->right;
    delete root1->left;
    delete root1;
    delete root3;
    
    return 0;
} 