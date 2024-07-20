#ifndef __VALUE_CORE_H__
#define __VALUE_CORE_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef RELU_ALPHA
#define RELU_ALPHA 0.01
#endif

void __value_add_backward__(__value_t* v);
void __value_sub_backward__(__value_t* v);
void __value_mul_backward__(__value_t* v);
void __value_div_backward__(__value_t* v);
void __value_pow_backward__(__value_t* v);
void __value_leaky_relu_backward__(__value_t* v);

__value_t* __value_add(__value_t* v_l, __value_t* v_r);
__value_t* __value_sub(__value_t* v_l, __value_t* v_r);
__value_t* __value_mul(__value_t* v_l, __value_t* v_r);
__value_t* __value_div(__value_t* v_l, __value_t* v_r);
__value_t* __value_pow(__value_t* v_l, __value_t* v_r);
__value_t* __value_leaky_relu(__value_t* v);

#ifdef AUTOGRAD_IMPLEMENTATION

/////////////////////////////////////
// Backward Util Functions
/////////////////////////////////////

void __value_add_backward__(__value_t* v) {
  v->depends[0]->grad += v->grad;
  v->depends[1]->grad += v->grad;

  __value_grad_clip(v->depends[0], MIN_GRAD, MAX_GRAD);
  __value_grad_clip(v->depends[1], MIN_GRAD, MAX_GRAD);
}

void __value_sub_backward__(__value_t* v) {
  v->depends[0]->grad += v->grad;
  v->depends[1]->grad -= v->grad;

  __value_grad_clip(v->depends[0], MIN_GRAD, MAX_GRAD);
  __value_grad_clip(v->depends[1], MIN_GRAD, MAX_GRAD);
}

void __value_mul_backward__(__value_t* v) {
  v->depends[0]->grad += v->depends[1]->val * v->grad;
  v->depends[1]->grad += v->depends[0]->val * v->grad;

  __value_grad_clip(v->depends[0], MIN_GRAD, MAX_GRAD);
  __value_grad_clip(v->depends[1], MIN_GRAD, MAX_GRAD);
}

void __value_div_backward__(__value_t* v) {
	v->depends[0]->grad += (1.0 / v->depends[1]->val) * v->grad;
	v->depends[1]->grad += (-v->depends[0]->val / (v->depends[1]->val * v->depends[1]->val)) * v->grad;

  __value_grad_clip(v->depends[0], MIN_GRAD, MAX_GRAD);
  __value_grad_clip(v->depends[1], MIN_GRAD, MAX_GRAD);
}

void __value_pow_backward__(__value_t* v) {
	v->depends[0]->grad += (v->depends[1]->val * pow(v->depends[0]->val, v->depends[1]->val - 1)) * v->grad;
	if (v->depends[0]->val > 0) {  // Ensure base is positive before computing log
		v->depends[1]->grad += (log(v->depends[0]->val) * pow(v->depends[0]->val, v->depends[1]->val)) * v->grad;
	}

  __value_grad_clip(v->depends[0], MIN_GRAD, MAX_GRAD);
  __value_grad_clip(v->depends[1], MIN_GRAD, MAX_GRAD);
}

void __value_leaky_relu_backward__(__value_t* v) {
	if (v->depends[0]->val > 0) {
		v->depends[0]->grad += v->grad;
	} else {
		v->depends[0]->grad += v->grad * RELU_ALPHA;
	}

  __value_grad_clip(v->depends[0], MIN_GRAD, MAX_GRAD);
}

/////////////////////////////////////
// Backward Functions
/////////////////////////////////////

__value_t* __value_add(__value_t* v_l, __value_t* v_r) {
	__value_t* v_out = __value_malloc(v_l->val + v_r->val);
	v_out->grad = 0;

	v_out->depends = (__value_t**)malloc(2 * sizeof(__value_t*));
	v_out->depends[0] = v_l;
	v_out->depends[1] = v_r;
	v_out->n_depend = 2;

	v_out->backward = __value_add_backward__;
	return v_out;
}

__value_t* __value_sub(__value_t* v_l, __value_t* v_r) {
	__value_t* v_out = __value_malloc(v_l->val - v_r->val);
	v_out->grad = 0;

	v_out->depends = (__value_t**)malloc(2 * sizeof(__value_t*));
	v_out->depends[0] = v_l;
	v_out->depends[1] = v_r;
	v_out->n_depend = 2;

	v_out->backward = __value_sub_backward__;
	return v_out;
}

__value_t* __value_mul(__value_t* v_l, __value_t* v_r) {
	__value_t* v_out = __value_malloc(v_l->val * v_r->val);
	v_out->grad = 0;

	v_out->depends = (__value_t**)malloc(2 * sizeof(__value_t*));
	v_out->depends[0] = v_l;
	v_out->depends[1] = v_r;
	v_out->n_depend = 2;

	v_out->backward = __value_mul_backward__;
	return v_out;
}

__value_t* __value_div(__value_t* v_l, __value_t* v_r) {
	if (v_r->val == 0.0) {
		printf("Error: Division by zero\n");
		exit(1);
	}

	__value_t* v_out = __value_malloc(v_l->val / v_r->val);
	v_out->grad = 0;

	v_out->depends = (__value_t**)malloc(2 * sizeof(__value_t*));
	v_out->depends[0] = v_l;
	v_out->depends[1] = v_r;
	v_out->n_depend = 2;

	v_out->backward = __value_div_backward__;
	return v_out;
}

__value_t* __value_pow(__value_t* v_l, __value_t* v_r) {
	__value_t* v_out = __value_malloc(pow(v_l->val, v_r->val));
	v_out->grad = 0;

	v_out->depends = (__value_t**)malloc(2 * sizeof(__value_t*));
	v_out->depends[0] = v_l;
	v_out->depends[1] = v_r;
	v_out->n_depend = 2;

	v_out->backward = __value_pow_backward__;
	return v_out;
}

__value_t* __value_leaky_relu(__value_t* v) {
	__value_t* v_out = __value_malloc(v->val);
	v->grad = 0;

	if (v->val <= 0) {
		v_out->val *= RELU_ALPHA;
	}

	v_out->grad = 0;
	v_out->depends = (__value_t**)malloc(sizeof(__value_t*));
	v_out->depends[0] = v;
	v_out->n_depend = 1;
	v_out->backward = __value_leaky_relu_backward__;

	return v_out;
}

#endif

#endif // __VALUE_CORE_H__
