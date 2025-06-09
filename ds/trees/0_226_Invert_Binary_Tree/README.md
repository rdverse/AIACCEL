# Invert Binary Tree

## LeetCode Question
[226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/description/)

Given the root of a binary tree, invert the tree, and return its root.

### Example 1:
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

```
Before Inversion:        After Inversion:
      4                       4
    /   \                   /   \
   2     7                 7     2
  / \   / \               / \   / \
 1   3 6   9             9   6 3   1
```

### Example 2:
Input: root = [2,1,3]
Output: [2,3,1]

```
Before Inversion:        After Inversion:
      2                       2
     / \                     / \
    1   3                   3   1
```

### Example 3:
Input: root = []
Output: []

```
Before Inversion:        After Inversion:
    (empty tree)           (empty tree)
```

### Constraints:
- The number of nodes in the tree is in the range [0, 100].
- -100 <= Node.val <= 100

## Solution

### Approach 1: Iterative DFS
- Use stack for DFS traversal
- For each node, swap its left and right children
- Push children to stack if they exist

```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;
        stack<TreeNode*> s;
        s.push(root);
        while (!s.empty()) {
            TreeNode* current = s.top();
            s.pop();
            swap(current->left, current->right);
            if (current->left) s.push(current->left);
            if (current->right) s.push(current->right);
        }
        return root;
    }
};
```

### Approach 2: Recursive DFS
- Use recursion to traverse the tree
- For each node, swap its left and right children
- Recursively invert left and right subtrees

```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;
        
        // Swap left and right children
        swap(root->left, root->right);
        
        // Recursively invert subtrees
        invertTree(root->left);
        invertTree(root->right);
        
        return root;
    }
};
```

### Complexity
- Time: O(n), where n is the number of nodes
- Space: 
  - Iterative: O(n) for the stack
  - Recursive: O(h) for recursion stack, where h is height of tree 