# Maximum Depth of Binary Tree

## LeetCode Question
[104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

### Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 3

```
Example Tree:
      3
     / \
    9   20
        / \
       15  7

Maximum Depth = 3 (path: 3 -> 20 -> 15 or 3 -> 20 -> 7)
```

### Example 2:
Input: root = [1,null,2]
Output: 2

```
Example Tree:
      1
       \
        2

Maximum Depth = 2 (path: 1 -> 2)
```

### Constraints:
- The number of nodes in the tree is in the range [0, 10^4].
- -100 <= Node.val <= 100

## Solution

### Approach 1: Recursive DFS
- Use recursion to traverse the tree
- For each node, return 1 + max(depth of left subtree, depth of right subtree)
- Base case: return 0 for null nodes

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};
```

### Approach 2: Iterative BFS
- Use a queue for level-order traversal
- Keep track of the current level
- Increment level counter for each level processed

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        
        queue<TreeNode*> q;
        q.push(root);
        int depth = 0;
        
        while (!q.empty()) {
            int levelSize = q.size();
            depth++;
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode* current = q.front();
                q.pop();
                
                if (current->left) q.push(current->left);
                if (current->right) q.push(current->right);
            }
        }
        return depth;
    }
};
```

### Approach 3: Iterative DFS
- Use a stack to simulate recursion
- Keep track of both node and its depth
- Update maximum depth when reaching leaf nodes

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        
        stack<pair<TreeNode*, int>> s;
        s.push({root, 1});
        int maxDepth = 0;
        
        while (!s.empty()) {
            auto [node, depth] = s.top();
            s.pop();
            
            maxDepth = max(maxDepth, depth);
            
            if (node->right) s.push({node->right, depth + 1});
            if (node->left) s.push({node->left, depth + 1});
        }
        return maxDepth;
    }
};
```

### Complexity Analysis
- Time Complexity:
  - Recursive DFS: O(n), where n is the number of nodes
  - Iterative BFS: O(n), where n is the number of nodes
  - Iterative DFS: O(n), where n is the number of nodes

- Space Complexity:
  - Recursive DFS: O(h), where h is the height of the tree (recursion stack)
  - Iterative BFS: O(w), where w is the maximum width of the tree (queue)
  - Iterative DFS: O(h), where h is the height of the tree (stack) 