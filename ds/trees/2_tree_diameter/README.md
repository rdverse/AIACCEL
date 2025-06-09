# Diameter of Binary Tree

## Problem
Find the length of the longest path between any two nodes in a binary tree.

## Naive Approach
- For every node, compute the height of left and right subtrees separately and update the maximum diameter.
- This repeats height calculations and is O(n^2).
- Time: O(n^2), Space: O(h), where n = number of nodes, h = height of tree

## Better Approach
- For each node, compute the height of left and right subtrees.
- The diameter at that node is left height + right height.
- Track the maximum diameter found.

## Complexity
- Time: O(n), where n is the number of nodes
- Space: O(h), where h is the height of the tree (recursion stack) 