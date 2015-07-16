Tutorial of mshadow
=====
This is a beginner's tutorial of mshadow. If you like mshadow and have ideas to improve this tutorial, you are more than welcomed:)
Please send a pull-request if you would like to share your experience.

See also other related materials about mshadow
* [Expression Template Tutorial](exp-template)
* [Writing Multi-GPU and Distributed ML](mshadow-ps)

**List of Topics**
* [Tensor Data Structure](#tensor-data-structure)
* [Memory Allocation](#memory-allocation)
* [Elementwise Operations](#elementwise-operations)
* [One code for both CPU and GPU](#one-code-for-both-cpu-and-gpu)
* [Matrix Multiplications](#matrix-multiplications)
* [User Defined Operator](#user-defined-operator)

Tensor Data Structure
====
The basic data structure of mshadow is Tensor. The following is a simplified equivalent version of
the declaration in [mashadow/tensor.h](../mshadow/tensor.h)
```c++
typedef unsigned index_t;
template<int dimension>
struct Shape {
  index_t shape_[dimension];
};
template<typename Device, int dimension, typename DType = float>
struct Tensor {
  DType *dptr_;
  Shape<dimension> shape_;
  Stream<Device> stream_;
  index_t stride_;
};
// this is how shape object declaration look like
Shape<2> shape2;
// this is how tensor object declaration look like
// you can
Tensor<cpu, 2> ts2;
Tensor<gpu, 3, float> ts3;
```
``` Tensor<cpu,2>``` means a two dimensional tensor in CPU, while ``` Tensor<gpu,3>``` means three dimensional tensor in GPU.
```Shape<k>``` gives the shape information of k-dimensional tensor. The declaration use template, and
can be specialized into tensor of specific device and dimension. This is what two dimensional tensor will look like:
```c++
struct Shape<2> {
  index_t shape_[2];
};
struct Tensor<cpu, 2, float> {
  float *dptr_;
  Shape<2> shape_;
  index_t stride_;
};
```
* ``` Tensor<cpu, 2>``` contains ```dptr_```, which points to the space that backup the tensor.
* ```Shape<2>``` is a structure that stores shape information, the convention is same as numpy
* ```stride_``` gives the number of cell space allocated in the smallest dimension (if we use numpy convention, the dimension corresponds to shape_[-1]).
  This is introduced when we introduce some padding cells in lowest dimension to make sure memory is aligned.
  - ```stride_``` is automatically set during memory allocation of tensor in mshadow.

To understand the data structure, consider the following code:
``` c++
float data[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
Tensor<cpu, 2> ts;
ts.dptr_ = data;
ts.shape_ = mshadow::Shape2(3, 2);
ts.stride_ = 3;
// now: ts[0][0] == 0, ts[0][1] == 1 , ts[1][0] == 3, ts[1][1] == 4
for (index_t i = 0; i < ts.size(0); ++i) {
  for (index_t j = 0; j < ts.size(1), ++j) {
    printf("ts[%u][%u]=%f\n", i, j, ts[i][j]);
  }
}
```
The result ts should be a 3 * 2 matrix, where data[2], data[5], data[8] are padding cells that are ignored. If you want a continuous memory, set ```stride_=shape_[1]```.

NOTICE: We highly recommend use stream in ```gpu``` mode, there will be an error thrown out if no stream is set. Check [basic_stream.cu](basic_stream.cu) for more detail.

Memory Allocation
====
An important design choice about mshadow is that the data structure is a **whitebox**:
it works so long as we set the space pointer ```dptr_```, corresponding ```shape_``` and ```stride_```:
* For ```Tensor<cpu, k>```, the space can be created by ```new float[]```, or pointer to some existing space such as float array in last example.
* For ```Tensor<gpu, k>```, the space need to lie in GPU, created by ```cudaMallocPitch```

mshadow also provide explicit memory allocation routine, demonstrated shown by following code
``` c++
// create a 5 x 3 tensor on GPU, and allocate space
Tensor<gpu, 2> ts2(Shape2(5, 3));
AllocSpace(&ts2);
// allocate 5 x 3 x 2 tensor on CPU, initialized by 0
Tensor<cpu, 3> ts3 = NewTensor<cpu>(Shape3(5,3,2), 0.0f);
// free space
FreeSpace(&ts2); FreeSpace(&ts3);
```
All memory allocations in mshadow are **explicit**. There is **no** implicit memory allocation and de-allocation during any operations.
This means ```Tensor<cpu, k>``` variable is more like a reference handle(pointer), instead of a object. If we assign a tensor to another variable, the two share the same content space.

This also allows user to use mshadow in their existing project easily, simply give mshadow the pointer of the memory and you can get the benefit of all the mshadow expressions with zero cost:)

We also have STL style container object called ```TensorContainer```, they behave exactly the same as Tensors, but the memory will be automatically freed during destruction.

Elementwise Operations
====
All the operators(+, -, *, /, += etc.) in mshadow are element-wise. Consider the following SGD update code:
```c++
void UpdateSGD(Tensor<cpu, 2> weight, Tensor<cpu, 2> grad, float eta, float lambda) {
  weight -= eta * (grad + lambda * weight);
}
```
During compilation, this code will be translated to the following form:
```c++
void UpdateSGD(Tensor<cpu,2> weight, Tensor<cpu,2> grad, float eta, float lambda) {
  for (index_t y = 0; y < weight.size(0); ++y) {
    for (index_t x = 0; x < weight.size(1); ++x) {
      weight[y][x] -= eta * (grad[y][x] + lambda * weight[y][x]);
    }
  }
}
```
As we can see, *no memory allocation* is happened in the translated code. For ```Tensor<gpu, k>```, the corresponding function will be translated into a CUDA kernel of same spirit.
Using [Expression Template](exp-template), the translation is happened during compile time. We can write simple lines of code while get the full performance of the translated code.

One code for both CPU and GPU
====
Since mshadow have identical interface for ```Tensor<cpu, k>``` and ```Tensor<gpu, k>```, we can easily write one code that works in both CPU and GPU.
For example, the following code compiles for both GPU and CPU Tensors.
```c++
template<typename xpu>
void UpdateSGD(Tensor<xpu, 2> weight, const Tensor<xpu, 2> &grad,
               float eta, float lambda) {
  weight -= eta * (grad + lambda * weight);
}
```
Matrix Multiplications
====
We also have short hands for dot product, as like follows. The code will be translated to call standard packages such as MKL and CuBLAS.
```c++
template<typename xpu>
void Backprop(Tensor<xpu, 2> gradin,
              const Tensor<xpu, 2> &gradout,
              const Tensor<xpu, 2> &netweight) {
  gradin = dot(gradout, netweight.T());
}
```
Again, the code can compile for both GPU and CPU Tensors

User Defined Operator
====
There are common cases when we want to define our own function. For example, assume we do not have element-wise sigmoid transformation in mshadow,
which is very commonly used in machine learning algorithms. We simply use the following code to add sigmoid to mshadow
```c++
struct sigmoid {
  MSHADOW_XINLINE static float Map(float a) {
    return 1.0f / (1.0f + expf(-a));
  }
};
template<typename xpu>
void ExampleSigmoid(Tensor<xpu, 2> out, const Tensor<xpu, 2> &in) {
  out = F<sigmoid>(in * 2.0f) + 1.0f;
}
```
The equivalent translated code for CPU is given by
```c++
template<typename xpu>
void ExampleSigmoid(Tensor<xpu, 2> out, const Tensor<xpu, 2> &in) {
  for (index_t y = 0; y < out.size(0); ++y) {
    for(index_t x = 0; x < out.size(1); ++x) {
      out[y][x] = sigmoid::Map(in[y][x] * 2.0f) + 1.0f;
    }
  }
}
```
Also note that the defined operation can be **composited into expressions**, not only we can write ```out = F<sigmoid>(in)```,
we can also write ```out = F<sigmoid>+2.0``` or ```out = F<sigmoid>(F<sigmoid>(in))```.

There will also be a translated CUDA kernel version that runs in GPU. Check out [defop.cpp](defop.cpp) for complete example.

Complete Example
====
The following code is from [basic.cpp](basic.cpp), that illustrate basic usage of mshadow.

```c++
// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

int main(void) {
  // intialize tensor engine before using tensor operation, needed for CuBLAS
  InitTensorEngine<cpu>();
  // assume we have a float space
  float data[20];
  // create a 2 x 5 x 2 tensor, from existing space
  Tensor<cpu, 3> ts(data, Shape3(2,5,2));
    // take first subscript of the tensor
  Tensor<cpu, 2> mat = ts[0];
  // Tensor object is only a handle, assignment means they have same data content
  // we can specify content type of a Tensor, if not specified, it is float bydefault
  Tensor<cpu, 2, float> mat2 = mat;

  // shaape of matrix, note size order is same as numpy
  printf("%u X %u matrix\n", mat.size(1), mat.size(1));

  // initialize all element to zero
  mat = 0.0f;
  // assign some values
  mat[0][1] = 1.0f; mat[1][0] = 2.0f;
  // elementwise operations
  mat += (mat + 10.0f) / 10.0f + 2.0f;

  // print out matrix, note: mat2 and mat1 are handles(pointers)
  for (index_t i = 0; i < mat.size(0); ++i) {
    for (index_t j = 0; j < mat.size(1); ++j) {
      printf("%.2f ", mat2[i][j]);
    }
    printf("\n");
  }
  // shutdown tensor enigne after usage
  ShutdownTensorEngine<cpu>();
  return 0;
}
```

