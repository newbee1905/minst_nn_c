#ifndef __VALUE_UTILS_H__
#define __VALUE_UTILS_H__

#include <stdio.h>
#include <stdlib.h>

#ifndef MIN_GRAD
#define MIN_GRAD -20.0
#endif

#ifndef MAX_GRAD
#define MAX_GRAD 20.0
#endif

void __value_print(__value_t* v);
void __value_grad_clip(__value_t* v, float min_grad, float max_grad);

#ifdef AUTOGRAD_IMPLEMENTATION

void __value_print(__value_t* v) {
  printf("Value(val=%.2f, grad=%.2f)\n", v->val, v->grad);
}

void __value_grad_clip(__value_t* v, float min_grad, float max_grad) {
	if (v->grad < min_grad) {
		v->grad = min_grad;
	} else if (v->grad > max_grad) {
		v->grad = max_grad;
	}
}

#endif

#endif // __VALUE_UTILS_H__
