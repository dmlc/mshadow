mshadow: Matrix Shadow
======

MShadow is a lightweight CPU/GPU Matrix/Tensor Template Library in C++/CUDA. The goal of mshadow is to support ***efficient***,
***device invariant*** and ***simple*** tensor library for machine learning project that aims for both simplicity and performance.

* [Contributors](https://github.com/tqchen/mshadow/graphs/contributors)
* [Tutorial](guide)

Features
=====
* Efficient: all the expression you write will be lazily evaluated and compiled into optimized code in ***compile time***.
  - No temporal memory allocation will happen for expression you write
  - mshadow will generate specific kernel for every expression you write in compile time.
* Device invariant: you can write 
* Simple: mshadow allows you to write machine learning code using expressions.
  - For example: ```weight = -eta * (grad + lambda * weight); ```, ```gradin = dot(gradout, netweight.T());```
* Whitebox: put a float* into the Tensor struct and take the benefit of the package, no memory allocation is happened unless explicitly called
* Lightweight library: light amount of code to support frequently used functions in machine learning
* Extendable: user can write simple functions that plugs into mshadow and run on GPU/CPU, no experience in CUDA is required.


Related Projects
=====
* CXXNET: neural network implementation based on mshadow: https://github.com/antinucleon/cxxnet
