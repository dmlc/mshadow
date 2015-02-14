mshadow-ps
====
### Parameter Server Interface for GPU Tensor

mshadow-ps provides asynchronize parameter server interface for mshadow GPU/CPU Tensor.
This allows you to do ***multi-GPU*** and ***disrtibuted*** (deep) learning in
an ***easy*** and ***unified*** way.

####List of Resources
* [API Documentation](http://homes.cs.washington.edu/~tqchen/mshadow/doc/namespacemshadow_1_1ps.html)
* [Library Interface Header](../../mshadow-ps/ps.h)
* Tutorial in this page

Tutorial
====
Suppose that we are now implementing a Multi-GPU learning program.
One way to do that is through data parallelism. We can launch many
threads, with each thread compute gradient on one GPU, and aggregate
the statistics together.
However, the gradient synchronization step could be cost time, and in
many cases, we can do the computation in an smarter way, so that 
we ***overlaps the computation with the synchronization***.

mshadow-ps provides interface to do such synchronization in an easy way.
The following documents provides a way 

### Getting Sum from Multiple GPUs
We first get familiar with the interface of mshadow-ps. Through the following
program in [local_sum-inl.h](local_sum-inl.h). You can compile the program
by setup the [config.mk](config.mk) according to your computers's enviroment, and type make.

In the following program, each thread first does some computation locally, then tries to get the sum
of ```data``` through mshadow-ps interface.
There are four key functions in ```ISharedModel``` interface
* [InitKey](../../mshadow-ps/ps.h#L76) allocates a key to specific tensor shape
* [Push](../../mshadow-ps/ps.h#L100) pushes out the local data to the synchronization interface
  - The data pushed by different devices will be aggregated together by key
  - Push is an asynchronize call and returns immediately
* [PullReq](../../mshadow-ps/ps.h#L122) requests the result of synchronization to be copied back
  - In the local default case, the synchronized result is the sum of pushed data
  - mshadow-ps also support the weight update on server side, where the result of PullReq is the updated weight instead of sum of gradient
  - PullReq is also asynchronize
* [PullWait](../../mshadow-ps/ps.h#L87) wait until the pull request of corresponding key finishes

```c++
// this function is runed by specific thread
template<typename xpu>
inline void RunWorkerThread(int devid,
                            mshadow::ps::ISharedModel<xpu, float> *ps) {
  // initialize tensor engine
  mshadow::InitTensorEngine<xpu>(devid);
  mshadow::Stream<xpu> *stream  = mshadow::NewStream<xpu>();
  // allocate tensor on xpu
  mshadow::TensorContainer<xpu, 2> data(mshadow::Shape2(2, 3));
  // set the computation stream to the new allocated stream
  // this will make subsequent computation whose target is data
  // to use the stream, stream is needed for async execution in GPU
  data.set_stream(stream);
  // assume these operations sets the content of dataient
  data[0] = 1.0f;
  data[1] = devid + data[0];
  printf("dev%d: before sync, data:\n", devid);
  // use print to show result, do not call
  // print normally since Copy will block
  Print(data);
  printf("====================\n");
  // intiaialize the key, register the shape on parameter server
  ps->InitKey(data[0].shape_, 0, devid);
  ps->InitKey(data[1].shape_, 1, devid);
  // push data[0] out, for update, or aggregation
  // 0 is the key of the data, devid is the current device id
  ps->Push(data[0], 0, devid);
  // pull request is used to request the data to be copied back
  // once computation is done
  ps->PullReq(data[0], 0, devid);
  // computation can be done here..
  // the pull request handler will be overlapped with   
  // similar as previous call
  ps->Push(data[1], 1, devid);
  ps->PullReq(data[1], 1, devid);
  // more computation can be done here...
  // the computation will be overlapped 
  // PullWait will block until these request finishes
  ps->PullWait(0, devid);
  ps->PullWait(1, devid);
  printf("dev%d: after sync, data:\n", devid);
  // use print to show result, do not call
  // print normally since Copy will block
  Print(data);
  printf("====================\n");
  mshadow::DeleteStream(stream);
  mshadow::ShutdownTensorEngine<xpu>();
}

template<typename xpu>
inline int Run(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: device list\n"\
           "\tfor CPU the device list can be arbitrary\n"\
           "\tfor GPU the device list need to be actual device index\n");
    return 0;
  }
  // list of device ids
  std::vector<int> devs;
  // initialization
  for (int i = 1; i < argc; ++i) {
    // record the device id
    devs.push_back(atoi(argv[i]));
  }
  mshadow::ps::ISharedModel<xpu, float>
      *ps = mshadow::ps::CreateSharedModel<xpu, float>("local");
  // intiaialize the ps
  ps->Init(devs);  
  // use openmp to launch #devs threads
  #pragma omp parallel num_threads(devs.size())
  {
    int tid = omp_get_thread_num();
    RunWorkerThread<xpu>(devs[tid], ps);
  }
  delete ps;
  return 0;
}
```
In the above example, we did not do weight update on server side, so the synchronization result is
simply the sum of data on each device. The key property of this interface is that the Push and PullReq are asynchronize.
* We can call these two functions once the gradient is ready, and the mshadow-ps will do the data synchronization in the background.
* When we need the result of synchronization, we simply call PullWait to wait the synchronization task to finish.
* Such interface allows us to do additional computation between the Push/PullReq and PullWait

### A MultiGPU Neural Net
To get a more concrete understanding of the interface. We give an example of multi-GPU two layer neuralnet
in [../neuralnet/nnet_ps.cu](../neuralnet/nnet_ps.cu). The general idea is follows
* Push and PullReq is called once we get the gradient of certain layer
* PullWait is called before we do forward on that layer next time
* This creates a ***time lag*** between the backprop and next forward to that layer
  - mshadow-ps do synchronization concurrently with computations during the time lag
  - The time lag is big for latter layers, which also usually need more time to synchronize

There are several note of the mshadow-ps on the neural net code
* Callback function in PullReq
  - A callback function can be pass to PullReq to be called when the request complete
  - We place weight update in the callback to perform update when we get the gradient sum
* Computing stream
  - Due to GPU's programming model, we need to do computation on non-default stream
  - Use set_stream in mshadow tensors to set stream to computation stream
  - To report error when you did not use stream, you can compile with -DMSHADOW_FORCE_STREAM

We should note thate because the example runs on MNIST, which is an quite small dataset, you may not observe
speedup with multiple cards. However, you will find significant speedup when you run on other tasks.
The newest version of [cxxnet](https://github.com/antinucleon/cxxnet)

### Moving Parameter Update to the Server
In all the examples so far, we use mshadow-ps to get the aggregated sum of gradients, and update
weights locally on each GPU. For more advanced usage of mshadow-ps, we can move the weight update
to the server. The communication pattern is as follows
* Each thread still call Push to push out gradient
* The server will apply the update rule to update the weight
* Each thread call PullReq to pull back the weight from server

Such update pattern is suitable under distributed setting. To do so, user need to implement an
[IModelUpdater](../../mshadow-ps/ps.h#L202) interface. And define the following CreateModelUpdater function
in the program
```c++
namespace mshadow {
namespace ps {
template<>
IModelUpdater<float> *CreateModelUpdater() {
  return new MyModelUpdater();
}
}
}
```
Before calling ISharedModel.Init, user need to call ```ps->SetParam("update_on_server", "1")``` to set the update 
mode on the server side. If user uses distributed shared model, user must define ModelUpdater.
