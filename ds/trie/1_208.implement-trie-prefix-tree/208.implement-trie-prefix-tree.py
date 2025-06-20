#
# @lc app=leetcode id=208 lang=python3
#
# [208] Implement Trie (Prefix Tree)
#

# @lc code=start
class TrieNode:
    def __init__(self):
        self.children ={}
        self.end = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for character in word:
            if character not in curr.children:
                curr.children[character] = TrieNode()
            curr = curr.children[character]
        curr.end = True
            
    def search(self, word: str) -> bool:
        curr = self.root
        for character in word:
            if character not in curr.children:
                return False
            curr = curr.children[character]
                
        return curr.end


    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for character in prefix:
            if character not in curr.children:
                return False
            else:
                curr = curr.children[character]
        return True # O(p), O(1)

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
# @lc code=end

