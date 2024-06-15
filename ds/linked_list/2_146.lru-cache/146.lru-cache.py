#
# @lc app=leetcode id=146 lang=python3
#
# [146] LRU Cache
#

# @lc code=start
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity             # Maximum number of items the cache can hold
        self.cache = OrderedDict()          # Stores the cache items in access order

    def get(self, key: int) -> int:
        if not key in self.cache:           # If key not present, return -1
            return -1 
        self.cache.move_to_end(key)           # Mark key as recently used (move to end)
        return self.cache[key]               # Return the value for the key

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)         # If key exists, mark as recently used
        self.cache[key] = value                 # Insert or update the key-value pair
        if len(self.cache) > self.capacity:    # If over capacity,
            self.cache.popitem(False)           # Remove least recently used item (from front)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
# @lc code=end
