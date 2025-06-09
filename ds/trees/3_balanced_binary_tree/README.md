# Balanced Binary Tree

## Problem
Check if a binary tree is height-balanced (for every node, the heights of left and right subtrees differ by at most 1).

## Naive Approach
- For every node, check the height of left and right subtrees and verify the difference is at most 1.
- This repeats height calculations and is O(n^2).
- Time: O(n^2), Space: O(h), where n = number of nodes, h = height of tree

## Better Approach
- Use recursion to check the height and balance of each subtree.
- Return both height and balance status from each call.

## Complexity
- Time: O(n), where n is the number of nodes
- Space: O(h), where h is the height of the tree (recursion stack) 