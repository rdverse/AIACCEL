# [169. Majority Element](https://leetcode.com/problems/majority-element/description/?envType=study-plan-v2&envId=top-interview-150)

Given an integer array <code>nums</code> sorted in **non-decreasing order** , remove some duplicates <a href="https://en.wikipedia.org/wiki/In-place_algorithm" target="_blank">**in-place** </a> such that each unique element appears **at most twice** . The **relative order**  of the elements should be kept the **same** .

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the **first part**  of the array <code>nums</code>. More formally, if there are <code>k</code> elements after removing the duplicates, then the first <code>k</code> elements of <code>nums</code>should hold the final result. It does not matter what you leave beyond the first<code>k</code>elements.

Return <code>k</code> after placing the final result in the first <code>k</code> slots of <code>nums</code>.

Do **not**  allocate extra space for another array. You must do this by **modifying the input array <a href="https://en.wikipedia.org/wiki/In-place_algorithm" target="_blank">in-place</a>**  with O(1) extra memory.

**Custom Judge:** 

The judge will test your solution with the following code:

```
int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
```

If all assertions pass, then your solution will be **accepted** .

**Example 1:** 

```
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
```

**Example 2:** 

```
Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3,_,_]
Explanation: Your function should return k = 7, with the first seven elements of nums being 0, 0, 1, 1, 2, 3 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
```

**Constraints:** 

- <code>1 <= nums.length <= 3 * 10^4</code>
- <code>-10^4 <= nums[i] <= 10^4</code>
- <code>nums</code> is sorted in **non-decreasing**  order.# 169.majority-element
This folder contains the processed file: 169.majority-element.py