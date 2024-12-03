#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

void auto_throw(cudaError_t status) {
    if(status != cudaSuccess) {
        std::string message = "ERROR: '";
        message += cudaGetErrorString(status);
        message +="'\n";
        throw std::runtime_error(message);
    }
}

template<typename T>
class ProCon {

    T* array;
    int  size;
    int *iter;

    public:

    // Construction is funky on CUDA. To make construction
    // more explicit, it is represented by a static method.
    static ProCon<T> create(int size) {
        ProCon<T> result;
        auto_throw(cudaMallocManaged(&result.array,sizeof(T)*size));
        auto_throw(cudaMallocManaged(&result.iter, sizeof(int)));
        result.size = size;
        *result.iter = 0;
        return result;
    }

    __device__ void give(T data) {
        int index = atomicAdd(iter,1);
        if(index >= size){
            // Guard for the error message, so the message prints
            // only once after the capacity is exceeded
            if(index == size){
                printf("ERROR: ProCon capacity exceeded!\n");
            }
            return;
        }
        array[index] = data;
    }

    __device__
    bool take(T& data){
        int index = atomicAdd(iter,-1);
        if(index < 1){
            return false;
        }
        data = array[index-1];
        return true;
    }

    // Destruction is also weird on CUDA
    void destruct() {
        auto_throw(cudaFree(array));
        auto_throw(cudaFree(iter));
    }

};

__device__
bool check_prime(int n){
    if(n < 2){
        return false;
    }
    for(int i = 2; i < n; i++){
        if(n % i == 0){
            return false;
        }
    }
    return true;
}

__global__
void produce_range(int start, int end, ProCon<int> output_data){
    size_t thread_count = gridDim.x * blockDim.x;
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = thread_id; i < end; i += thread_count){
        output_data.give(i);
    }
}


__global__ 
void filter_for_primes(ProCon<int> input_data, ProCon<int> output_data) {
    int data;
    while(input_data.take(data)){
        if(check_prime(data)){
            // Process data
            output_data.give(data);
        }
    }
}

__global__ 
void print_procon(ProCon<int> input_data) {
    int data;
    while(input_data->take(data)){
        printf("%d\n",data);
    }
}

int main()
{
    size_t const CAPACITY = 10;
    ProCon<int>* input = ProCon<int>::create(CAPACITY);
    ProCon<int>* output = ProCon<int>::create(CAPACITY);

    // Fill input with data
    produce_range<<<1,1>>>(0,10,input);

    filter_for_primes<<<1,1>>>(input, output);

    // Read output
    print_procon<<<1,1>>>(output);

    input.destruct();
    output.destruct();
    return 0;
}