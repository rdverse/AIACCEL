/*
 * @lc app=leetcode id=146 lang=cpp
 *
 * [146] LRU Cache
 */

// @lc code=start
#include <unordered_map>

class LRUCache {
    struct Node{
        int key; // mapping to remove node
        int value;
        Node* left;
        Node* right;
        Node(int key, int value) : key(key), value(value){}
    };

    Node* head;
    Node* tail;
    std::unordered_map<int, Node*> cache;
    int capacity;

    void move_to_end(Node* curr){
        auto before = tail->left;
        before->right = curr;
        curr->left = before;
        curr->right = tail; 
        tail->left = curr; 
    }

    void rewire_pointers(Node* curr){
        auto before = curr->left;
        auto after = curr->right;
        before->right=after;
        after->left=before;
    }


    public:
    LRUCache(int capacity) : capacity(capacity){
       head = new Node(-1,-1);
       tail = new Node(-1,-1);
        head->right = tail;
        tail->left = head;
    }

    int get(int key) {
        if (!cache.contains(key)){
            return -1;
        }
        Node* curr = cache[key];
        int val = curr->value;
        rewire_pointers(curr);
        move_to_end(curr);
        return val;

    }
    
    void put(int key, int value) {


        if (cache.contains(key)){
            Node* curr = cache[key];
            rewire_pointers(curr);
            //curr->value = value;
            cache.erase(key);
        }
            Node* inserted = new Node(key, value);
            cache[key]=inserted;
            move_to_end(inserted);

            if (cache.size()> capacity){
                Node* curr = head->right;
                rewire_pointers(curr);
                //move_to_end(curr);
                cache.erase(curr->key);
                //delete curr;
        }
    }
};

// single linked list doesnt have ref to prev node
// list is expensive
// dll is a possible solution 

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
// @lc code=end