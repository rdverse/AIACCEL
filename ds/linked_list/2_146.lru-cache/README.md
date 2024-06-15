# [146. LRU Cache](https://leetcode.com/problems/lru-cache/description/?envType=study-plan-v2&envId=top-interview-150)

Design a data structure that follows the constraints of a **<a href="https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU" target="_blank">Least Recently Used (LRU) cache</a>** .

Implement the <code>LRUCache</code> class:

- <code>LRUCache(int capacity)</code> Initialize the LRU cache with **positive**  size <code>capacity</code>.
- <code>int get(int key)</code> Return the value of the <code>key</code> if the key exists, otherwise return <code>-1</code>.
- <code>void put(int key, int value)</code> Update the value of the <code>key</code> if the <code>key</code> exists. Otherwise, add the <code>key-value</code> pair to the cache. If the number of keys exceeds the <code>capacity</code> from this operation, **evict**  the least recently used key.

The functions <code>get</code> and <code>put</code> must each run in <code>O(1)</code> average time complexity.

**Example 1:** 

```
Input

["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output

[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation

LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
```

**Constraints:** 

- <code>1 <= capacity <= 3000</code>
- <code>0 <= key <= 10^4</code>
- <code>0 <= value <= 10^5</code>
- At most <code>2 * 10^5</code> calls will be made to <code>get</code> and <code>put</code>.