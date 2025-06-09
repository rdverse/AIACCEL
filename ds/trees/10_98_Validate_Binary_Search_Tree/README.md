# Validate Binary Search Tree

## LeetCode Question
[98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.

### Example 1:
Input: root = [2,1,3]
Output: true

```
Valid BST:
      2
     / \
    1   3
```

### Example 2:
Input: root = [5,1,4,null,null,3,6]
Output: false

```
Invalid BST:
      5
     / \
    1   4
        / \
       3   6

Explanation: The root node's value is 5 but its right child's value is 4.
```

### Example 3:
Input: root = [2,2,2]
Output: false

```
Invalid BST:
      2
     / \
    2   2

Explanation: Duplicate values are not allowed in BST.
```

### Constraints:
- The number of nodes in the tree is in the range [1, 10^4].
- -2^31 <= Node.val <= 2^31 - 1

## Solution Nodes

### Approach 1: Inorder Traversal
- Use inorder traversal (left -> root -> right)
- For a valid BST, inorder traversal should give sorted values
- Keep track of previous value and check if current value is greater
- Time Complexity: O(n)
- Space Complexity: O(h) where h is height of tree

### Approach 2: Range Check
- Pass valid range (min, max) to each node
- For each node, check if its value is within the range
- Update ranges for left and right subtrees
- Time Complexity: O(n)
- Space Complexity: O(h) where h is height of tree

### Approach 3: Iterative Inorder
- Use stack to simulate inorder traversal
- Keep track of previous value
- Check if values are strictly increasing
- Time Complexity: O(n)
- Space Complexity: O(h) where h is height of tree 