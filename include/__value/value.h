#ifndef __VALUE_VALUE_H__
#define __VALUE_VALUE_H__

#include <stdio.h>
#include <stdlib.h>

typedef struct __value __value_t;

struct __value {
  float val;
  float grad;
  __value_t** depends;
  int n_depend;
  void (*backward)(__value_t*);
};

__value_t* __value_alloc(float x);
__value_t** __value_arr_alloc(float* vals, size_t sz);
void __value_free(__value_t* v);

#ifdef AUTOGRAD_IMPLEMENTATION

__value_t* __value_alloc(float x) {
  __value_t* v = (__value_t*)malloc(sizeof(__value_t));
  if (!v) {
		perror("Memory allocation failed");
		exit(1);
  }

  v->val = x;
  v->grad = 0;

  v->depends = NULL;
  v->n_depend = 0;
  v->backward = NULL;

  return v;
}

__value_t** __value_arr_alloc(float* vals, size_t sz) {
	__value_t** v_arr = (__value_t**)malloc(sz * sizeof(__value_t*));
  if (v_arr == NULL) {
    perror("Memory allocation failed");
    exit(1);
  }

  for (size_t i = 0; i < sz; ++i) {
  	v_arr[i] = __value_alloc(vals[i]);
  }

  return v_arr;
}

void __value_free(__value_t* v) {
	if (!v) {
		return;
	}

	if (v->depends) {
		free(v->depends);
		v->depends = NULL;
	}
	free(v);
	v = NULL;
}

#endif

#endif // __VALUE_VALUE_H__
