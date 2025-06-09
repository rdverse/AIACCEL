#include <iostream>
#include <vector>
#include <climits>

using namespace std;

// Definition for a binary tree node.
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
    bool isValidBST(TreeNode* root) {
        return valid(root, LONG_MIN, LONG_MAX);
    }

    bool valid(TreeNode* node, long left, long right) {
        if (!node) {
            return true;
        }
        if (!(left < node->val && node->val < right)) {
            return false;
        }
        return valid(node->left, left, node->val) &&
               valid(node->right, node->val, right);
    }
}; // O(n), O(n)

// Helper function to create a binary tree from an array
TreeNode* createTree(const vector<int>& nums, int index) {
    if (index >= nums.size() || nums[index] == INT_MIN) {
        return nullptr;
    }
    
    TreeNode* root = new TreeNode(nums[index]);
    root->left = createTree(nums, 2 * index + 1);
    root->right = createTree(nums, 2 * index + 2);
    return root;
}

int main() {
    Solution solution;
    
    // Test cases with expected answers
    vector<vector<int>> testCases = {
        {2, 1, 3},           // Expected: true  (Valid BST: 1 < 2 < 3)
        {5, 1, 4, INT_MIN, INT_MIN, 3, 6},  // Expected: false (Invalid: 3 < 4 < 5, but 3 is in right subtree of 4)
        {2, 2, 2}            // Expected: false (Invalid: duplicates not allowed in BST)
    };
    
    for (const auto& testCase : testCases) {
        TreeNode* root = createTree(testCase, 0);
        cout << "Test case: [";
        for (int i = 0; i < testCase.size(); i++) {
            if (testCase[i] == INT_MIN) cout << "null";
            else cout << testCase[i];
            if (i < testCase.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        bool result = solution.isValidBST(root);
        cout << "Is valid BST: " << (result ? "true" : "false") << endl << endl;
    }
    
    return 0;
} 