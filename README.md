# MNIST for C

This repository contains a header file for loading MNIST dataset in C language.


## preparation

1. download the following data files

	- train image : `train-images.idx3-ubyte`
	- train label : `train-labels.idx1-ubyte`
	- test image : `t10k-images.idx3-ubyte`
	- test label : `t10k-labels.idx1-ubyte`  

	you can find these files inside `/data` directory in this repository  
	they are also available at http://yann.lecun.com/exdb/mnist/

2. download `mnist.h`

3. in `mnist.h`, set the appropriate data path to the following macros

	- `#define TRAIN_IMAGE (ex: "./data/train-images.idx3-ubyte")`
	- `#define TRAIN_LABEL (ex: "./data/train-labels.idx1-ubyte")`
	- `#define TEST_IMAGE  (ex: "./data/t10k-images.idx3-ubyte")`
	- `#define TEST_LABEL  (ex: "./data/t10k-labels.idx1-ubyte")`

4. add `#include "mnist.h"` in your C code


## usage

after calling the following *void* function,  
`load_mnist();`  
you will be able to access the corresponding data through the following array:

	train image : train_image[60000][784] (type: double, normalized, flattened)
	train label : train_label[60000]      (type: int)
	test image  : test_image[10000][784]  (type: double, normalized, flattened)
	test label  : test_label[10000]       (type: int)


## functions

- void **load_mnist (** void **)**  
	load mnist data to respective array (as mentioned above)  

- void **print_mnist_pixel (** double data_image[][], int num_data **)**  
	data_image[][]: `train_image` or `test_image`  
	num_data: `NUM_TRAIN` or `NUM_TEST`  

- void **print_mnist_label (** double data_label[][], int num_data **)**  
	data_label[][]: `train_image` or `test_image`  
	num_data: `NUM_TRAIN` or `NUM_TEST`  

- void **save_mnist_pgm (** double data_image[][SIZE], int index **)**  
	data_image[][]: `train_image` or `test_image`  
	index: index of data to save (`0~59999` for train image, `0~9999` for test image)  


## example

	#include "mnist.h"

	int main(void)
	{
		load_mnist();
		
		print_mnist_pixel(test_image, NUM_TEST);
		
		print_mnist_label(test_label, NUM_TEST);
		
		save_mnist_pgm(test_image, 0);
		
		return 0;
	}
