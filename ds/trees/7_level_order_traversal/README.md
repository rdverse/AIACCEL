# Binary Tree Level Order Traversal

## Problem
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

## Example
Input:
```
     3
    / \
   9  20
      / \
     15  7
```
Output: `[[3], [9,20], [15,7]]`

## Return Type
- Return a vector of vectors: `vector<vector<int>>`
- Each inner vector represents one level
- Nodes in each level are ordered from left to right

## Naive Approach
- Use a queue to store nodes at each level
- For each level:
  - Get the current size of queue (nodes at this level)
  - Process all nodes at current level
  - Add their children to queue for next level
- Time: O(n), Space: O(n) where n = number of nodes

## Better Approach
### BFS with Queue
- Use a queue to implement BFS
- Keep track of level size to separate levels
- Process nodes level by level
- Time: O(n), Space: O(n) for queue

### DFS with Level Tracking
- Use DFS with level parameter
- Store nodes in a map/vector based on their level
- Time: O(n), Space: O(n) for recursion stack and result

## Complexity
- Time: O(n), where n is the number of nodes
- Space: O(n) for queue/recursion stack and result storage 