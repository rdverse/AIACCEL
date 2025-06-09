#include <iostream>
#include <vector>
#include <climits>
#include <stack>

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

class Solution{
    public:
        int kthSmallest(TreeNode* root, int k){
            // O(n), O(n) considering skewness. Otherwise O(H+K) and O(H) best case
            std::stack<TreeNode*> nodes;
            
            nodes.push(root);

            TreeNode* curr  = root; 
            int kelement=-1;

            while(!nodes.empty()){
                while(curr->left){ // fill stage
                    nodes.push(curr->left);
                    curr = curr->left;
                    }
                  
                    TreeNode* next = nodes.top();
                    kelement = next->val;
                    nodes.pop();
                    k--;
                    if(k==0){
                        return kelement;   
                    }
                    if (next->right){
                    nodes.push(next->right); 
                    curr=next->right; 
                    }
                    
            }
            return -1;
        }
}; // O(n) and O(n) (for balanced, O(logn)/O(H)) and O(logn)/O(H)





int main() {
    Solution solution;
    
    // Test cases with expected answers
    vector<pair<vector<int>, int>> testCases = {
        {{3, 1, 4, INT_MIN, 2}, 1},  // Expected: 1 (1st smallest)
        {{5, 3, 6, 2, 4, INT_MIN, INT_MIN, 1}, 3}  // Expected: 3 (3rd smallest)
    };
    
    for (const auto& [treeArray, k] : testCases) {
        TreeNode* root = createTree(treeArray, 0);
        cout << "Test case: [";
        for (int i = 0; i < treeArray.size(); i++) {
            if (treeArray[i] == INT_MIN) cout << "null";
            else cout << treeArray[i];
            if (i < treeArray.size() - 1) cout << ", ";
        }
        cout << "], k = " << k << endl;
        
        int result = solution.kthSmallest(root, k);
        cout << k << "th smallest element: " << result << endl << endl;
    }
    
    return 0;
} 