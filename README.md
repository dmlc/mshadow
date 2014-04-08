mshadow: Matrix Shadow
======

Lightweight CPU/GPU Matrix/Tensor Template Library in C++/CUDA

Creater: Bing Xu and Tianqi Chen


Documentation and Tutorial: https://github.com/tqchen/mshadow/wiki

Description
=====
Most machine learning algorithms requires matrix,tensor operations frequently. For example, Eq.(1) is a common SGD update rule, where the weight can be a vector, matrix or 3D tensor. Eq.(2) is the backpropagtion rule:
```
(1) weight =  - eta * ( grad + lambda * weight ); 
(2) gradin = dot( gradout, netweight.T() );
```

These operations are not hard to implement, even in C++. The first one is elementwise operations, and can easily be written as 
```
for( int index = 0; index < weight.length; index ++ ){ 
  weight[index] = - eta * ( grad[index] + lambda * weight[index] ); 
}
```
Eq.(2) is matrix product, and we can use standard BLAS packages such as Intel MKL. It will looklike
```
sgemm( CblasNoTrans, CblasTrans, n, m, k, 1.0, gradout.ptr, lda, netweight.ptr, ldb, 0.0, gradin.ptr, ldc );
```
However:

* It is annoying to write these codes repeatively, and they are not intuitive. 
* What if we want to port our code to GPU? We need to rewrite our code in CUDA

mshadow is a unified C++/CUDA lib to to write Eq.(1) and Eq.(2) in C++, and *translate* them to the for loop and standard packages such as MKL, CuBLAS *in compile time*. 


Features
=====
* Shadow instead of giant: mshadow does not implement all of the functions,  it is more of a wrapper to translated easy-to-read code to standard 'giant' packages such as MKL
* Whitebox instead of blackbox: put a float* into the Tensor struct and take the benefit of the package, no memory allocation is happened unless explicitly called
* Unified CPU/GPU code: write a code and it should run in both CPU and GPU
* Lightweight library: light amount of code to support frequently used functions in machine learning
* Extendable: user can write simple functions that plugs into mshadow and run on GPU/CPU, no experience in CUDA is required.


Related Projects
=====
* CXXNET: neural network implementation based on mshadow: https://github.com/antinucleon/cxxnet
