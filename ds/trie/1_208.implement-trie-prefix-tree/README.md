# [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/description/?envType=study-plan-v2&envId=top-interview-150)

A <a href="https://en.wikipedia.org/wiki/Trie" target="_blank">**trie** </a> (pronounced as "try") or **prefix tree**  is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

- <code>Trie()</code> Initializes the trie object.
- <code>void insert(String word)</code> Inserts the string <code>word</code> into the trie.
- <code>boolean search(String word)</code> Returns <code>true</code> if the string <code>word</code> is in the trie (i.e., was inserted before), and <code>false</code> otherwise.
- <code>boolean startsWith(String prefix)</code> Returns <code>true</code> if there is a previously inserted string <code>word</code> that has the prefix <code>prefix</code>, and <code>false</code> otherwise.

**Example 1:** 

```
Input

["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output

[null, null, true, false, true, null, true]

Explanation

Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
```

**Constraints:** 

- <code>1 <= word.length, prefix.length <= 2000</code>
- <code>word</code> and <code>prefix</code> consist only of lowercase English letters.
- At most <code>3 * 10^4</code> calls **in total**  will be made to <code>insert</code>, <code>search</code>, and <code>startsWith</code>.