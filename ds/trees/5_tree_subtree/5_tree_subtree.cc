#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <tuple>

/*
Approach 1 (Naive):
- For each node in the main tree, check if the subtree starting from that node matches the given subtree
- Uses two functions:
  1. is_subtree: Recursively checks each node in the main tree
  2. is_sametree: Checks if two trees are identical
- Time Complexity: O(m*n) where m is nodes in main tree and n is nodes in subtree
- Space Complexity: O(h) where h is height of the main tree (recursion stack)

Approach 2 (KMP):
- Convert both trees to string representation using preorder traversal
- Use KMP algorithm to find if subtree string is a substring of main tree string
- Time Complexity: O(m+n) where m and n are lengths of the strings
- Space Complexity: O(m+n) for storing the strings
*/

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
        // Naive approach: Check each node in the main tree
        // For each node, check if the subtree starting from that node matches the given subtree
        bool is_subtree(TreeNode* root, TreeNode* subRoot) {
            if (!subRoot) {
                return true;  // Empty tree is a subtree of any tree
            }
            if (!root) {
                return false;  // Non-empty subtree can't be in an empty tree
            }
            if (is_sametree(root, subRoot)) {
                return true;  // Found a match at current node
            }
            return is_subtree(root->left, subRoot) || is_subtree(root->right, subRoot);  // Check children
        }

    private:
        // Helper function to check if two trees are identical
        bool is_sametree(TreeNode* p, TreeNode* q){
            if (!p && !q){
                return true;  // Both trees are empty
            } 
            if ( p && q && (p->val==q->val)){
                return (is_sametree(p->left, q->left) && is_sametree(p->right, q->right));  // Check children
            }
            return false;  // Trees don't match
        }
};


class Solution_kmp {
    public:
        // KMP approach: Convert trees to strings and use KMP algorithm
        // TODO: Implement string conversion and KMP algorithm
        bool is_subtree(TreeNode* root, TreeNode* subRoot) {
            return false;
        }
};


int main() {
    Solution naive;
    Solution_kmp kmp;
    
    // Test Case 1: Simple subtree
    TreeNode* tree1 = new TreeNode(3, 
        new TreeNode(4, new TreeNode(1), new TreeNode(2)), 
        new TreeNode(5));
    TreeNode* subtree1 = new TreeNode(4, new TreeNode(1), new TreeNode(2));
    std::cout << "Test Case 1 (Simple subtree):\n";
    std::cout << "Naive approach: " << naive.is_subtree(tree1, subtree1) << std::endl;
    std::cout << "KMP approach: " << kmp.is_subtree(tree1, subtree1) << std::endl;
    
    // Test Case 2: Not a subtree
    TreeNode* tree2 = new TreeNode(3, 
        new TreeNode(4, new TreeNode(1), new TreeNode(2, new TreeNode(0), nullptr)), 
        new TreeNode(5));
    TreeNode* subtree2 = new TreeNode(4, new TreeNode(1), new TreeNode(2));
    std::cout << "\nTest Case 2 (Not a subtree):\n";
    std::cout << "Naive approach: " << naive.is_subtree(tree2, subtree2) << std::endl;
    std::cout << "KMP approach: " << kmp.is_subtree(tree2, subtree2) << std::endl;
    
    // Test Case 3: Empty subtree
    TreeNode* tree3 = new TreeNode(1, new TreeNode(2), new TreeNode(3));
    TreeNode* subtree3 = nullptr;
    std::cout << "\nTest Case 3 (Empty subtree):\n";
    std::cout << "Naive approach: " << naive.is_subtree(tree3, subtree3) << std::endl;
    std::cout << "KMP approach: " << kmp.is_subtree(tree3, subtree3) << std::endl;
    
    // Test Case 4: Complex subtree
    TreeNode* tree4 = new TreeNode(1, 
        new TreeNode(2, new TreeNode(4), new TreeNode(5)), 
        new TreeNode(3, new TreeNode(6), new TreeNode(7)));
    TreeNode* subtree4 = new TreeNode(2, new TreeNode(4), new TreeNode(5));
    std::cout << "\nTest Case 4 (Complex subtree):\n";
    std::cout << "Naive approach: " << naive.is_subtree(tree4, subtree4) << std::endl;
    std::cout << "KMP approach: " << kmp.is_subtree(tree4, subtree4) << std::endl;

    return 0;
}
