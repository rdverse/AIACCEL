#include <iostream>
using namespace std;

//* Definition for a binary tree node.

/*
Time and space complexity:
time - O(n) 
space - O(n), best O(logn) balanced tree

 */

struct TreeNode {
      int val;
      TreeNode *left;
      TreeNode *right;
      TreeNode() : val(0), left(nullptr), right(nullptr) {}
      TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
      TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
  };
 
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        // If both nodes are null, trees are same
        if (p == NULL && q == NULL) {
            return true;
        }
        
        // If one node is null and other isn't, trees are different
        if (p == NULL || q == NULL || p->val != q->val) {
            return false;
        }

        // pval and qval is same so check the next val
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};



int main() {
    Solution solution;
    
    // Test Case 1: Simple same trees
    TreeNode* tree1 = new TreeNode(1, new TreeNode(2), new TreeNode(3));
    TreeNode* tree2 = new TreeNode(1, new TreeNode(2), new TreeNode(3));
    cout << "Test Case 1 (Simple same trees): " << solution.isSameTree(tree1, tree2) << endl;
    
    // Test Case 2: Different values
    TreeNode* tree3 = new TreeNode(1, new TreeNode(2), new TreeNode(4));
    cout << "Test Case 2 (Different values): " << solution.isSameTree(tree1, tree3) << endl;
    
    // Test Case 3: Different structure
    TreeNode* tree4 = new TreeNode(1, new TreeNode(2, new TreeNode(4), nullptr), new TreeNode(3));
    cout << "Test Case 3 (Different structure): " << solution.isSameTree(tree1, tree4) << endl;
    
    // Test Case 4: Complex same trees
    TreeNode* tree5 = new TreeNode(1, 
        new TreeNode(2, new TreeNode(4), new TreeNode(5)), 
        new TreeNode(3, new TreeNode(6), new TreeNode(7)));
    TreeNode* tree6 = new TreeNode(1, 
        new TreeNode(2, new TreeNode(4), new TreeNode(5)), 
        new TreeNode(3, new TreeNode(6), new TreeNode(7)));
    cout << "Test Case 4 (Complex same trees): " << solution.isSameTree(tree5, tree6) << endl;
    
    // Test Case 4: Empty trees
    TreeNode* root5 = nullptr;
    TreeNode* root6 = nullptr;
    
    cout << "Test Case 4 (Empty trees): " << solution.isSameTree(root5, root6) << endl;
    
    return 0;
}