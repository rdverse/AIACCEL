// imports
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>


/*
Given the root of a binary tree, return its depth.


The depth of a binary tree is defined as the number of nodes
along the longest path from the root node down to the farthest leaf node.


Sol : use recursion, find height of subtree
Time : O(n) and O(logn) for balanced
Space : O(h)

Alternate methods : DFS (STack) , BFS (Queue) - Better for large trees, scope for parallelization
*/

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
};

int find_height(TreeNode* node){
    if (!node){return 0;} 
    else{
    return(1 + std::max(find_height(node->left), find_height(node->right)));
    }
} // Time and Space : O(n), O(n)

int main(){
    TreeNode* root = new TreeNode(1);
    TreeNode* tree1 = new TreeNode(1, new TreeNode(2), new TreeNode(3)); //height 2
    TreeNode* tree2 = new TreeNode(1, new TreeNode(2, new TreeNode(4), new TreeNode(5)), new TreeNode(3, new TreeNode(6), new TreeNode(7))); // height 3
    TreeNode* tree3 = new TreeNode(1, new TreeNode(2, new TreeNode(4, new TreeNode(8), new TreeNode(9)), new TreeNode(5, new TreeNode(10), new TreeNode(11))), new TreeNode(3, new TreeNode(6, new TreeNode(12), new TreeNode(13)), new TreeNode(7, new TreeNode(14), new TreeNode(15)))); // height 4

    std::vector<TreeNode*> trees = {tree1, tree2, tree3};
    
    for(auto tree : trees){
        std::cout << "Height of tree: " << find_height(tree) << std::endl;
    }

    return 0;
}


/*
tree1
    1
   / \
  2   3

tree2
       1
     /   \
    2     3
   / \   / \
  4  5  6  7

tree3
              1
           /     \
         2         3
       /  \       /  \
     4     5     6    7
    /\    /\    /\    /\
   8  9 10 11 12 13 14 15

 */