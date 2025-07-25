# [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/?envType=study-plan-v2&envId=top-interview-150)

Given a string <code>s</code>, find the length of the **longest**  <button type="button" aria-haspopup="dialog" aria-expanded="false" aria-controls="radix-:rj:" data-state="closed" class="">**substring** </button> without duplicate characters.

**Example 1:** 

```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
```

**Example 2:** 

```
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
```

**Example 3:** 

```
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

**Constraints:** 

- <code>0 <= s.length <= 5 * 10^4</code>
- <code>s</code> consists of English letters, digits, symbols and spaces.