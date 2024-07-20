#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOPO_MAX_SIZE 100001

#define AUTOGRAD_IMPLEMENTATION
#include "autograd.h"

#define TENSOR_IMPLEMENTATION
#include "tensor.h"

signed main() {
	float vals[] = {2.0, 3.0, 4.0};
	__value_t** v_arr = __value_arr_alloc(vals, 3);

	__value_t* sum1 = __value_pow(v_arr[0], v_arr[1]);
	__value_t* sum2 = __value_div(v_arr[2], v_arr[0]);
	__value_t* sum = __value_leaky_relu(__value_sub(__value_alloc(0), __value_mul(sum1, sum2)));

	__value_backward(sum);

	for (size_t i = 0; i < 3; ++i) {
		__value_print(v_arr[i]);
	}
	__value_print(sum1);
	__value_print(sum2);
	__value_print(sum);

	for (size_t i = 0; i < 3; ++i) {
		__value_free(v_arr[i]);
	}
	free(v_arr);
	__value_free(sum1);
	__value_free(sum2);
	__value_free(sum);
}
