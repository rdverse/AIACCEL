#
# @lc app=leetcode id=380 lang=python3
#
# [380] Insert Delete GetRandom O(1)
#
import random
# @lc code=start
class RandomizedSet:
    # {"a" : 1, "b":2, "c":3}
    def __init__(self):
        self.values = []
        self.dictionary = {}

    def insert(self, val: int) -> bool:
        if val in self.dictionary:
            return False
        self.dictionary[val] = len(self.values)
        self.values.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val in self.dictionary:
            # swap dictionary's position in -1 with idx
            # loc_list : it is the postiion of the val in list | serves as a unique value for a key in the dict 
            # dict
            loc_list = self.dictionary[val]
            # list
            last_value = self.values[-1]
            self.values[loc_list] = last_value # duplicate val - pop
            self.values.pop()
            ## 
            del self.dictionary[val]
            if last_value in self.dictionary:
                self.dictionary[last_value] = loc_list 
            return True
        else:
            return False 
                 
    def getRandom(self) -> int:
        return random.choice(self.values)
    # O(1) time
    
    
# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
# @lc code=end

