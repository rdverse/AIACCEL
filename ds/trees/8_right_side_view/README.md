# Binary Tree Right Side View

## Problem
Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

## Example
Input:
```
     1
    / \
   2   3
    \   \
     5   4
```
Output: `[1, 3, 4]`

## Return Type
- Return a vector of integers: `vector<int>`
- Each integer represents the rightmost node value at that level
- Order is from top to bottom

## Approaches

### BFS with Queue
- Use level order traversal
- For each level, take the last node (rightmost)
- Time: O(n), Space: O(n)

### DFS with Level Tracking
- Use DFS with level parameter
- Keep track of the rightmost node at each level
- Time: O(n), Space: O(h) where h is height of tree

## Complexity
- Time: O(n), where n is the number of nodes
- Space: 
  - BFS: O(n) for queue
  - DFS: O(h) for recursion stack, where h is height of tree 