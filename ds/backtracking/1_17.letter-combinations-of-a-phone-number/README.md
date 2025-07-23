# [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/?envType=study-plan-v2&envId=top-interview-150)

Given a string containing digits from <code>2-9</code> inclusive, return all possible letter combinations that the number could represent. Return the answer in **any order** .

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
<img alt="" src="https://assets.leetcode.com/uploads/2022/03/15/1200px-telephone-keypad2svg.png" style="width: 300px; height: 243px;">

**Example 1:** 

```
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**Example 2:** 

```
Input: digits = ""
Output: []
```

**Example 3:** 

```
Input: digits = "2"
Output: ["a","b","c"]
```

**Constraints:** 

- <code>0 <= digits.length <= 4</code>
- <code>digits[i]</code> is a digit in the range <code>['2', '9']</code>.