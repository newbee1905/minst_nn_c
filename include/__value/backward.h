#ifndef __VALUE_BACKWARD_H__
#define __VALUE_BACKWARD_H__

#include <stdio.h>
#include <stdlib.h>

void __value_topo_init(__value_t* v_root, __value_t** topo, size_t* topo_sz, __value_t** visited, size_t* visited_sz);
void __value_backward(__value_t* v_root);

#ifdef AUTOGRAD_IMPLEMENTATION

void __value_topo_init(__value_t* v_root, __value_t** topo, size_t* topo_sz, __value_t** visited, size_t* visited_sz) {
	for (size_t i = 0; i < *visited_sz; ++i) {
		if (visited[i] == v_root) {
			return;
		}
	}

	visited[(*visited_sz)++] = v_root;

	for (size_t i = 0; i < v_root->n_depend; ++i) {
		__value_topo_init(v_root->depends[i], topo, topo_sz, visited, visited_sz);
	}

	topo[(*topo_sz)++] = v_root;
}

void __value_backward(__value_t* v_root) {
	__value_t* topo[TOPO_MAX_SIZE];
	size_t topo_sz = 0;
	__value_t* visited[TOPO_MAX_SIZE];
	size_t visited_sz = 0;

	__value_topo_init(v_root, topo, &topo_sz, visited, &visited_sz);

	v_root->grad = 1.0;

	for (size_t i = topo_sz; i > 0; --i) {
		if (topo[i - 1]->backward) {
			topo[i - 1]->backward(topo[i - 1]);
		}
	}
}

#endif

#endif // __VALUE_BACKWARD_H__
