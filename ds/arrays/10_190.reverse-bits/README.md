# [190. Reverse Bits](https://leetcode.com/problems/reverse-bits/description/?source=submission-noac)

Given an integer array <code>nums</code>, rotate the array to the right by <code>k</code> steps, where <code>k</code> is non-negative.

**Example 1:** 

```
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
```

**Example 2:** 

```
Input: nums = [-1,-100,3,99], k = 2
Output: [3,99,-1,-100]
Explanation: 
rotate 1 steps to the right: [99,-1,-100,3]
rotate 2 steps to the right: [3,99,-1,-100]
```

**Constraints:** 

- <code>1 <= nums.length <= 10^5</code>
- <code>-2^31 <= nums[i] <= 2^31 - 1</code>
- <code>0 <= k <= 10^5</code>

**Follow up:** 

- Try to come up with as many solutions as you can. There are at least **three**  different ways to solve this problem.
- Could you do it in-place with <code>O(1)</code> extra space?