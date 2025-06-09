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

// TODO: Implement your solution here
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    TreeNode* curr = root;
    
//    if(curr->val > std::min(p->val,q->val)){ // atleast one of p/q is in right subtree

    while(curr){
   
        if((p->val < curr->val) && (q->val < curr->val)){
            curr = curr->left;
        } else if ((p->val > curr->val) && (q->val > curr->val)) {
            curr = curr->right;
        }
        else{
            return curr;
        }
    }
    return nullptr;
} // O(h), O(1)

TreeNode* lowestCommonAncestorRecursive(TreeNode* root, TreeNode* p, TreeNode* q) {
    //    if(curr->val > std::min(p->val,q->val)){ // atleast one of p/q is in right subtree
        if((p->val < root->val) && (q->val < root->val)){
            return lowestCommonAncestorRecursive(root->left, p, q);
        } else if ((p->val > root->val) && (q->val > root->val)) {
            return lowestCommonAncestorRecursive(root->right, p, q);
        }
        else{
            return root;
        }
    return nullptr;
} // O(h) iterations, O(h) for calls


int main() {
    // Test Case 1: Simple BST
    //       6
    //      / \
    //     2   8
    //    / \ / \
    //   0  4 7  9
    //     / \
    //    3   5
    TreeNode* root1 = new TreeNode(6,
        new TreeNode(2,
            new TreeNode(0),
            new TreeNode(4,
                new TreeNode(3),
                new TreeNode(5))),
        new TreeNode(8,
            new TreeNode(7),
            new TreeNode(9)));
    std::cout << "answers : 4, 6, 2";
    TreeNode* p1 = root1->left->right->left;  // Node 3
    TreeNode* q1 = root1->left->right->right; // Node 5
    std::cout << "Test Case 1 (LCA of 3 and 5): " << lowestCommonAncestor(root1, p1, q1)->val << std::endl;
    std::cout << "Test Case 1 Recursive (LCA of 3 and 5): " << lowestCommonAncestorRecursive(root1, p1, q1)->val << std::endl<< std::endl;

    // Test Case 2: LCA is root
    TreeNode* p2 = root1->left->left;  // Node 0
    TreeNode* q2 = root1->right->right; // Node 9
    std::cout << "Test Case 2 (LCA of 0 and 9): " << lowestCommonAncestor(root1, p2, q2)->val << std::endl;
    std::cout << "Test Case 2 Recursive (LCA of 0 and 9): " << lowestCommonAncestorRecursive(root1, p2, q2)->val << std::endl<< std::endl;
    
    // Test Case 3: One node is ancestor of other
    TreeNode* p3 = root1->left;  // Node 2
    TreeNode* q3 = root1->left->right->left; // Node 3
    std::cout << "Test Case 3 (LCA of 2 and 3): " << lowestCommonAncestor(root1, p3, q3)->val << std::endl;
    std::cout << "Test Case 3 Recursive (LCA of 2 and 3): " << lowestCommonAncestorRecursive(root1, p3, q3)->val << std::endl<< std::endl;
    
    return 0;
} 