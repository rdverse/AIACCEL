# Lowest Common Ancestor in Binary Search Tree

## Problem
Given a binary search tree (BST) where all node values are unique, and two nodes from the tree p and q, return the lowest common ancestor (LCA) of the two nodes.

The lowest common ancestor between two nodes p and q is the lowest node in a tree T such that both p and q as descendants. The ancestor is allowed to be a descendant of itself.

## Naive Approach
- For each node, check if both p and q are in its left or right subtree.
- If they are in different subtrees, current node is LCA.
- Time: O(n), Space: O(h), where n = number of nodes, h = height of tree

## Better Approach
### Iterative Solution
- Use BST property: left subtree < root < right subtree
- If p and q are both less than root, go left
- If p and q are both greater than root, go right
- Otherwise, current node is LCA
- Time: O(h), Space: O(1) where h is height of tree

### Recursive Solution
- Same logic as iterative but using recursion
- If p and q are both less than root, recurse left
- If p and q are both greater than root, recurse right
- Otherwise, return current node
- Time: O(h), Space: O(h) for recursion stack

## Complexity
- Time: O(h), where h is the height of the tree
- Space: O(1) for iterative, O(h) for recursive 