#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "autograd.h"

typedef struct {
	__value_t **data;
	size_t *strides;
	size_t *shape;

	size_t ndim;
	size_t size;
	// char *device;
} tensor_t;

tensor_t* tensor_alloc(float* data, size_t* shape, size_t ndim);
__value_t* tensor_get(tensor_t* t, size_t* indices);

#define TENSOR_IMPLEMENTATION
#ifdef TENSOR_IMPLEMENTATION

tensor_t *tensor_alloc(float *data, size_t *shape, size_t ndim) {
	tensor_t *t = (tensor_t*)malloc(sizeof(tensor_t));
	if (!t) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(1);
	}

	t->shape = shape;
	t->ndim = ndim;

	t->size = 1;
	for (size_t i = 0; i < ndim; ++i) {
		t->size *= shape[i];
	}

	t->data = __value_arr_alloc(data, t->size);

	t->strides = (size_t*)malloc(ndim * sizeof(size_t));
	if (!t->strides) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(1);
	}

	size_t stride = 1;
	for (size_t i = ndim; i > 0; --i) {
		t->strides[i - 1] = stride;
		stride *= shape[i - 1];
	}
}

__value_t* tensor_get(tensor_t* t, size_t* indices) {
	size_t index = 0;
	for (int i = 0; i < t->ndim; index += indices[i] * t->strides[i], ++i); 

	return t->data[index];
}

#endif

#endif // __TENSOR_H__