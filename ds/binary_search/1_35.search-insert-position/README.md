# [35. Search Insert Position](https://leetcode.com/problems/search-insert-position/description/?envType=study-plan-v2&envId=top-interview-150)

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You mustwrite an algorithm with<code>O(log n)</code> runtime complexity.

**Example 1:** 

```
Input: nums = [1,3,5,6], target = 5
Output: 2
```

**Example 2:** 

```
Input: nums = [1,3,5,6], target = 2
Output: 1
```

**Example 3:** 

```
Input: nums = [1,3,5,6], target = 7
Output: 4
```

**Constraints:** 

- <code>1 <= nums.length <= 10^4</code>
- <code>-10^4 <= nums[i] <= 10^4</code>
- <code>nums</code> contains **distinct**  values sorted in **ascending**  order.
- <code>-10^4 <= target <= 10^4</code>