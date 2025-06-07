#include <stdio.h>
#include <iostream>
#include<c>



struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
};




// height_balanced lh == rh +-1

class sol{
    public:
        bool is_balanced(TreeNode* node){
            unsigned int* difference = 0; // >1 not balanced 
            find_height(node, &difference);
        }

    private:
        int find_height(TreeNode* node, int *diff){ //height

            int left_height  = find_height(node->left, diff);
            
            int right_height = find_height(node->right, diff);

            diff = std::static_cast<unsigned int>(left_height - right_height);   

            return(1 + std::max(left_height, right_height))

        }
};

// node h(l), h(r), return False - if any of them are not equivalent

int main(){
    
}