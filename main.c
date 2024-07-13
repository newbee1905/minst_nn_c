#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MIN_GRAD -10.0
#define MAX_GRAD 10.0

typedef struct __value __value_t;

struct __value {
  float val;
  float grad;
  __value_t** depends;
  int n_depend;
  void (*backward)(__value_t*);
};

__value_t* __value_init(float x) {
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

__value_t** __value_arr_init(float* vals, size_t sz) {
	__value_t** v_arr = (__value_t**)malloc(sz * sizeof(__value_t*));
  if (v_arr == NULL) {
    perror("Memory allocation failed");
    exit(1);
  }

  for (size_t i = 0; i < sz; ++i) {
  	v_arr[i] = __value_init(vals[i]);
  }

  return v_arr;
}

void __print_value(__value_t* v) {
  printf("Value(val=%.2f, grad=%.2f)\n", v->val, v->grad);
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

void __value_grad_clip(__value_t* v, float min_grad, float max_grad) {
	if (v->grad < min_grad) {
		v->grad = min_grad;
	} else if (v->grad > max_grad) {
		v->grad = max_grad;
	}
}

void __value_add_backward__(__value_t* v) {
  v->depends[0]->grad += v->grad;
  v->depends[1]->grad += v->grad;
  __value_grad_clip(v->depends[0], MIN_GRAD, MAX_GRAD);
  __value_grad_clip(v->depends[1], MIN_GRAD, MAX_GRAD);
}

__value_t* __value_add(__value_t* v_l, __value_t* v_r) {
	__value_t* v = __value_init(v_l->val + v_r->val);
	v->grad = 0;

	v->depends = (__value_t**)malloc(2 * sizeof(__value_t*));
	v->depends[0] = v_l;
	v->depends[1] = v_r;
	v->n_depend = 2;

	v->backward = __value_add_backward__;
	return v;
}

signed main() {
	float vals[] = {1.0, 2.0, 3.0};
	__value_t** v_arr = __value_arr_init(vals, 3);

	__value_t* sum = __value_add(v_arr[0], v_arr[1]);
	sum->grad = 1.0;

	sum->backward(sum);

	for (size_t i = 0; i < 3; ++i) {
		__print_value(v_arr[i]);
	}
	__print_value(sum);

	for (size_t i = 0; i < 3; ++i) {
		__value_free(v_arr[i]);
	}
	free(v_arr);
	__value_free(sum);
}
