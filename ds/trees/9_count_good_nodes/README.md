# Count Good Nodes in Binary Tree

## Problem
Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.

Return the number of good nodes in the binary tree.

## Example
Input:
```
     3
    / \
   1   4
  /   / \
 3   1   5
```
Output: `4`
Explanation:
- Node 3 (root) is always a good node
- Node 4 is a good node (3 < 4)
- Node 5 is a good node (3 < 4 < 5)
- Node 3 (leaf) is a good node (3 = 3)
- Node 1 is not a good node (3 > 1)

## Return Type
- Return an integer: `int`
- The count of good nodes in the tree

## Approaches

### DFS with Stack (Iterative)
- Use stack to implement DFS
- Store node and max value in path as pair
- If current node value >= max, increment count
- Time: O(n), Space: O(n)

### DFS with Recursion
- Use recursive DFS
- Pass max value in path as parameter
- Return count of good nodes in subtree
- Time: O(n), Space: O(h) where h is height of tree

### BFS with Queue
- Use queue to implement BFS
- Store node and max value in path as pair
- If current node value >= max, increment count
- Time: O(n), Space: O(n)

## Complexity
- Time: O(n), where n is the number of nodes
- Space: 
  - DFS (Stack): O(n) for stack
  - DFS (Recursion): O(h) for recursion stack
  - BFS: O(n) for queue 