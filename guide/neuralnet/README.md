This folder contains a mshadow example of simple neural net implementation

To compile the code, type make:
* You will need to have CUDA and MKL installed.
* Alternatively, you can compile with CBLAS packages to replace MKL such as BLAS or ATLAS, type make blas=1

To run the demo, download  MNIST dataset from: http://yann.lecun.com/exdb/mnist/
unzip all the files into current folder

and run by  ./nnet cpu or ./nnet gpu. ./convnet cpu or ./convnet gpu
