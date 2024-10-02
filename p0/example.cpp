#include <iostream>
#include <cmath>
#include <omp.h>
#include <cstdio>

bool naive_prime_check(int value){
    int limit = sqrt(value) + 1;
    for(int i=2; i<=limit; i++){
        if( (value%i) == 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]){

    omp_set_num_threads( 4 );

    int const VAL_COUNT = 100;

    std::cout << "The list of primes below "
              << VAL_COUNT << " is:\n\n";

    #pragma omp parallel for
    for(int i=0; i<VAL_COUNT; i++){
        if(naive_prime_check(i)){
            // Using printf to avoid race conditions
            // and other shenanagins.
            printf("%d\n",i);
            // The printed values may still be out of
            // order, but they should always be
            // separated by a newline
        }
    }

    return 0;

}
