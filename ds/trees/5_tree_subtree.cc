#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <tuple>

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
        bool is_subtree(TreeNode* root, TreeNode* subRoot) {
            if (!subRoot) {
                return true;
            }
            if (!root) {
                return false;
            }
            if (is_sametree(root, subRoot)) {
                return true;
            }
            return is_subtree(root->left, subRoot) || is_subtree(root->right, subRoot);
        }

    private:
        
        // given a node, tell weather the trees match
        bool is_sametree(TreeNode* p, TreeNode* q){
            if (!p && !q){
                return true;
            } 
            if ( p && q && (p->val==q->val)){
                return (is_sametree(p->left, q->left) && is_sametree(p->right, q->right));
                }
            return false;
        }
};


class Solution_kmp {
    public:
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
