#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <vector>
/*
bottom up, 
return height, isbalanced, node
*/

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
};

// height_balanced lh == rh +-1
class sol{
    public:
        bool is_balanced(TreeNode* node){
            //int difference; // >1 not balanced 
            //bool balanced;
            //int height;
            //brute_find_height(node, &difference);
            //return difference<2 ? true : false;
            auto [height, balanced] = dfs(node);
            //std::vector<int> num  = dfs(node);
            printf("Got height and balanced as %d %d\n", height, balanced);
            //printf("Got height and balacnded as %d %d", num[0], num[1]);
            return balanced;
        }

    private:

        // int brute_find_height(TreeNode* node, int *diff){ //height
        //     int left_height  = brute_find_height(node->left, diff);
        //     int right_height = brute_find_height(node->right, diff);
        //     *diff = left_height - right_height;   
        //     return(1 + std::max(left_height, right_height));
        // }


    // std::vector<int> dfs(TreeNode* root) {
    //     if (!root) {
    //         return {1, 0};
    //     }

    //     std::vector<int> left = dfs(root->left);
    //     std::vector<int> right = dfs(root->right);

    //     bool balanced = (left[0] == 1 && right[0] == 1) && 
    //                     (abs(left[1] - right[1]) <= 1);
    //     int height = 1 + std::max(left[1], right[1]);

    //     return {balanced ? 1 : 0, height};
    // }
        // t
        std::tuple<int, bool> dfs(TreeNode* node) {
            if (node == nullptr) {
                return {0, true};  // Use curly braces, not std::tuple()
            }

            auto [left_height, left_balanced] = dfs(node->left);
            auto [right_height, right_balanced] = dfs(node->right);
            
            int curr_height = 1 + std::max(left_height, right_height);
            
            if ((right_balanced && left_balanced) && (abs(right_height - left_height) <= 1)) { 
                return {curr_height, true};  // Use curly braces
            } else {
                return {curr_height, false};  // Use curly braces
            }
        }
};

// node h(l), h(r), return False - if any of them are not equivalent

int main() {
    sol solution;
    
    // Test Case 1: Simple balanced tree
    TreeNode* tree1 = new TreeNode(1, new TreeNode(2), new TreeNode(3));
    std::cout << "Test Case 1 (Simple balanced tree): " << solution.is_balanced(tree1) << std::endl;
    
    // Test Case 2: Unbalanced tree (left heavy)
    TreeNode* tree2 = new TreeNode(1, 
        new TreeNode(2, new TreeNode(3), nullptr), 
        nullptr);
    std::cout << "Test Case 2 (Left heavy balanced): " << solution.is_balanced(tree2) << std::endl;
    
    // Test Case 3: Unbalanced tree (right heavy)
    TreeNode* tree3 = new TreeNode(1, 
        nullptr, 
        new TreeNode(2, 
            nullptr, 
            new TreeNode(3, 
                nullptr, 
                new TreeNode(4, 
                    nullptr, 
                    new TreeNode(5)))));
    std::cout << "Test Case 3 (Right heavy unbalanced): " << solution.is_balanced(tree3) << std::endl;
    
    // Test Case 4: Complex balanced tree
    TreeNode* tree4 = new TreeNode(1, 
        new TreeNode(2, new TreeNode(4), new TreeNode(5)), 
        new TreeNode(3, new TreeNode(6), new TreeNode(7)));
    std::cout << "Test Case 4 (Complex balanced tree): " << solution.is_balanced(tree4) << std::endl;
    
    // Test Case 5: Empty tree
    TreeNode* tree5 = nullptr;
    std::cout << "Test Case 5 (Empty tree): " << solution.is_balanced(tree5) << std::endl;
    
    // Cleanup
    delete tree1;
    delete tree2;
    delete tree3;
    delete tree4;
    
    return 0;
}