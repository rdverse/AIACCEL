# [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=top-interview-150)

Given an integer array <code>nums</code>, return an array <code>answer</code> such that <code>answer[i]</code> is equal to the product of all the elements of <code>nums</code> except <code>nums[i]</code>.

The product of any prefix or suffix of <code>nums</code> is **guaranteed**  to fit in a **32-bit**  integer.

You must write an algorithm that runs in<code>O(n)</code>time and without using the division operation.

**Example 1:** 

```
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
```

**Example 2:** 

```
Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
```

**Constraints:** 

- <code>2 <= nums.length <= 10^5</code>
- <code>-30 <= nums[i] <= 30</code>
- The input is generated such that <code>answer[i]</code> is **guaranteed**  to fit in a **32-bit**  integer.

**Follow up:** Can you solve the problem in <code>O(1)</code>extraspace complexity? (The output array **does not**  count as extra space for space complexity analysis.)