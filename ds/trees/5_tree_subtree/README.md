# Subtree of Another Tree

## Problem
Check if one binary tree is a subtree of another.

## Naive Approach
- For each node in the main tree, check if the subtree starting from that node matches the given subtree using a tree comparison function.
- Time: O(m*n), Space: O(h), where m = nodes in main tree, n = nodes in subtree, h = height of main tree

## Better Approach
- For each node in the main tree, check if the subtree starting from that node matches the given subtree (naive approach).
- Optionally, use string serialization and KMP for optimization.

## Complexity
- Naive: Time O(m*n), Space O(h), where m = nodes in main tree, n = nodes in subtree, h = height of main tree
- KMP (TODO): Time O(m+n), Space O(m+n) (for string conversion) 