#include<stdio.h>
#include<iostream>
#include<queue>
#include<stack>

struct node{
    int val;
    node *left;
    node *right;
    node(int val): val(val), left(nullptr), right(nullptr) {}
    node(int val, node *left, node *right): val(val), left(left), right(right) {}
    node() {val = 0; left = nullptr; right = nullptr;}
};



class solution {
    public:
    node* solution_dfs_inverted(node*root){
        if (!root) return nullptr;
        std::stack<node*> s;
        s.push(root);
        while (!s.empty()){
            node* current = s.top();
            s.pop();
            //printf("%d ", current->val);
            _swap(current->left, current->right);
            if (current->left) s.push(current->left);
            if (current->right) s.push(current->right);
        }
        return root;
    } // iterative dfs

node* solution_bfs(node* root) {
    if (!root) return nullptr;
    std::queue<node*> q;
    q.push(root); 
    while (!q.empty()){
        node* current = q.front();
        q.pop();
        printf("%d ", current->val);
        if (current->left) q.push(current->left);
        if (current->right) q.push(current->right);
    }
    return root;

} // levelorder - bfs

void _swap(node*& left, node*& right){
    node* temp = left;
    left = right;
    right = temp;
}

    // node* simple_sol(node* root){
    //      swap(root->left, root->right);
    //      simple_sol(root->left);
    //      simple_sol(root->right);
    //     return root;
    // }

}; //dfs


int main(){

// create some small trees
node *tree1 = new node(1, new node(2), new node(3));
node *tree2 = new node(1, new node(2, new node(4), new node(5)), new node(3, new node(6), new node(7)));

solution sol;
// print the trees
printf("bfs non-inverted: ");
sol.solution_bfs(tree1);
printf("\n");
sol.solution_bfs(tree2);
printf("\n\n\n'");

printf("dfs inverted: ");
node* sol1 = sol.solution_dfs_inverted(tree1);
sol.solution_bfs(sol1);
printf("\n");
node* sol2 = sol.solution_dfs_inverted(tree2);
sol.solution_bfs(sol2);


return 0;
}