#include <stdio.h>
#include <iostream>
#include <vector>



/*
The diameter of a binary tree is defined as the length 
of the longest path between any two nodes within the tree.
 The path does not necessarily have to pass through the root.

The length of a path between two nodes in a binary tree is the number of edges between the nodes.

Given the root of a binary tree root, return the diameter of the tree.

Input: root = [1,null,2,3,4,5]

Output: 3

Input: root = [1,2,3]

Output: 2 

Time and Space : O(n), O(h)

Alternate solutions : 1. Brute force, for each height - find max height for each node O(n^2), O(n)
2. 




*/


struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
};


// bfs
// cur = n nodes in a level
// res = max(res, current)
class Diameter_Tree{

    public:

        int find_diameter(TreeNode* root){
            int result=0;
            dfs(root, result);
            return result;
}

    private:
        int dfs(TreeNode* node, int &result){
        
        if(!node){return 0;}

        int left_diameter = dfs(node->left, result);
        int right_diameter = dfs(node->right, result);
        result = std::max(result,left_diameter+right_diameter); // global update to result
        return (1 + std::max(left_diameter, right_diameter)); // return height
        }
};


int main(){
    TreeNode* root = new TreeNode(1);
    TreeNode* tree1 = new TreeNode(1, new TreeNode(2), new TreeNode(3)); //diameter 2
    TreeNode* tree2 = new TreeNode(1, new TreeNode(2, new TreeNode(4), new TreeNode(5)), new TreeNode(3, new TreeNode(6), new TreeNode(7))); // diameter 4 
    TreeNode* tree3 = new TreeNode(1, new TreeNode(2, new TreeNode(4, new TreeNode(8), new TreeNode(9)), new TreeNode(5, new TreeNode(10), new TreeNode(11))), new TreeNode(3, new TreeNode(6, new TreeNode(12), new TreeNode(13)), new TreeNode(7, new TreeNode(14), new TreeNode(15)))); // diameter 8 

    std::vector<TreeNode*> trees = {tree1, tree2, tree3};

    for (TreeNode* tree : trees){
        Diameter_Tree* sol;
       std::cout<<"Diameter of tree : " << sol->find_diameter(tree)<< std::endl; 
    }
}