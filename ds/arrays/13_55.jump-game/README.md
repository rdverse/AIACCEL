# [55. Jump Game](https://leetcode.com/problems/jump-game/?envType=study-plan-v2&envId=top-interview-150)

You are given an integer array <code>nums</code>. You are initially positioned at the array's **first index** , and each element in the array represents your maximum jump length at that position.

Return <code>true</code> if you can reach the last index, or <code>false</code> otherwise.

**Example 1:** 

```
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:** 

```
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
```

**Constraints:** 

- <code>1 <= nums.length <= 10^4</code>
- <code>0 <= nums[i] <= 10^5</code>