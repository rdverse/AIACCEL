# Same Binary Tree

## Problem
Check if two binary trees are identical (structure and node values).

## Naive Approach
- Recursively compare each node and its children in both trees.
- Time: O(n), Space: O(h), where n = number of nodes (minimum of both trees), h = height of tree

## Better Approach
- Recursively compare the root, left, and right subtrees of both trees.
- If all match, the trees are the same.

## Complexity
- Time: O(n), where n is the number of nodes (minimum of both trees)
- Space: O(h), where h is the height of the tree (recursion stack) 