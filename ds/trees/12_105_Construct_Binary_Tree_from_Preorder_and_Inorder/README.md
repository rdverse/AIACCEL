# Construct Binary Tree from Preorder and Inorder Traversal

## LeetCode Question
[105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

### Example 1:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

```
Constructed Tree:
      3
     / \
    9   20
        / \
       15  7

Explanation:
1. First element in preorder (3) is the root
2. In inorder, elements before 3 (9) are in left subtree
3. Elements after 3 (15,20,7) are in right subtree
4. For left subtree:
   - Next element in preorder (9) is root
   - No elements before 9 in inorder, so no left children
   - No elements after 9 in inorder, so no right children
5. For right subtree:
   - Next element in preorder (20) is root
   - Elements before 20 (15) are in left subtree
   - Elements after 20 (7) are in right subtree
```

### Example 2:
Input: preorder = [-1], inorder = [-1]
Output: [-1]

```
Constructed Tree:
    -1

Explanation: Single node tree
```

### Example 3:
Input: preorder = [1,2], inorder = [2,1]
Output: [1,2]

```
Constructed Tree:
    1
   /
  2

Explanation:
1. First element in preorder (1) is the root
2. In inorder, elements before 1 (2) are in left subtree
3. No elements after 1 in inorder, so no right subtree
```

### Constraints:
- 1 <= preorder.length <= 3000
- inorder.length == preorder.length
- -3000 <= preorder[i], inorder[i] <= 3000
- preorder and inorder consist of unique values
- Each value of inorder also appears in preorder
- inorder is guaranteed to be the inorder traversal of the tree
- preorder is guaranteed to be the preorder traversal of the tree

## Solution Nodes

### Approach 1: Single-Pass DFS with Index Tracking
- Use two indices (preId, inId) to track progress in both arrays
- Use a limit value to determine subtree boundaries
- Build tree in a single pass through both arrays
- Key points:
  - preId tracks current root in preorder
  - inId tracks current position in inorder
  - limit value helps determine when to stop building subtrees
  - No need for hash map or array splitting

#### Algorithm Steps:
1. Start with preId = 0, inId = 0
2. For each node:
   - Create root from preorder[preId]
   - If inorder[inId] == limit, return null (subtree boundary)
   - Recursively build left subtree with current root as limit
   - Recursively build right subtree with original limit

#### Complexity:
- Time Complexity: O(n) - single pass through both arrays
- Space Complexity: O(h) - recursion stack height, where h is tree height

### Approach 2: Hash Map with Array Splitting
- Use hash map to store inorder indices for O(1) lookup
- Split arrays into left and right subtrees
- More intuitive but less space efficient

#### Algorithm Steps:
1. Create hash map of inorder values to indices
2. For each node:
   - First element in preorder is root
   - Find root's position in inorder using hash map
   - Split inorder array into left and right subtrees
   - Recursively build subtrees with corresponding array portions

#### Complexity:
- Time Complexity: O(n) - each node is processed once
- Space Complexity: O(n) - for hash map and recursion stack