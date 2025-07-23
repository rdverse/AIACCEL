#include<stdio.h>
#include<pthread.h>
#include<vector>
/*
Row-wise tensor parallelism
Input                [scatter]                       [triad kernel]            [all-reduce]
A(mxn) -> A1(mxn1) , A2(mxn2), ...A4(mxn4) ->          Ai*Wi+Ci           ->      B(mxn)
                n1+n2+n3+n4=n                   Ai(mx1),Wi(nx1),Ci(mx1)       
*/

// Structure to hold thread data
typedef struct {
    int thread_id;
    double* local_data;
    double* result;
    int data_size;
} thread_data_t;

// all_reduce
/*
In all reduce, each gpu gets a copy of data
after forward, all intermediate tensors are averaged and returned back to the gpu
this code will simulate that process
*/

class collectives{
    public:
    virtual void* producer(){
    }
    virtual void* consumer(){
    }
    ~collectives(){}
};

class all_reduce : public collectives{
    public:
    virtual void* producer(){
    }
    virtual void* consumer(){
    }
};

class scatter : public collectives{
    public:
    virtual void* producer(){
    }
    virtual void* consumer(){
    }
};


std::vector<int> triad_kernel(std::vector<int> Act, std::vector<int> Wei, std::vector<int> C){
    
}




int main(){
return 0;
}