# Kth Smallest Element in a BST

## LeetCode Question
[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

Given the root of a binary search tree and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

### Example 1:
Input: root = [3,1,4,null,2], k = 1
Output: 1

```
BST:
      3
     / \
    1   4
     \
      2

Explanation: The 1st smallest element is 1.
```

### Example 2:
Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3

```
BST:
          5
         / \
        3   6
       / \
      2   4
     /
    1

Explanation: The 3rd smallest element is 3.
```

### Constraints:
- The number of nodes in the tree is n.
- 1 <= k <= n <= 10^4
- 0 <= Node.val <= 10^4

## Solution Nodes

### Approach 1: Inorder Traversal
- Use inorder traversal (left -> root -> right)
- Keep track of count of nodes visited
- Return when count equals k
- Time Complexity: O(n)
- Space Complexity: O(h) where h is height of tree

### Approach 2: Iterative Inorder
- Use stack to simulate inorder traversal
- Keep track of count of nodes visited
- Return when count equals k
- Time Complexity: O(n)
- Space Complexity: O(h) where h is height of tree

### Approach 3: Augmented Tree
- Store count of nodes in left subtree for each node
- Use this information to find kth smallest element
- Time Complexity: O(h) where h is height of tree
- Space Complexity: O(n) for storing counts 