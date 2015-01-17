#include "./ps.h"
using namespace mshadow;
void Print1DTensor(Tensor<cpu, 1, float> const &ts) {
  for (index_t i = 0; i < ts.size(0); ++i) {
    printf("%.2f ", ts[i]);
  }
  printf("\n");
}

void Print2DTensor(Tensor<cpu, 2, float> const &ts) {
  for (index_t i = 0; i < ts.size(0); ++i) {
    Print1DTensor(ts[i]);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage:<ndev>\n"); return 0;
  } 
  int ndev = atoi(argv[1]);
  ps::IParamServer<gpu, float> *ps = ps::Create<gpu, float>("local");
  TensorContainer<gpu, 3, float> ts(Shape3(ndev,5,2));
  TensorContainer<gpu, 3, float> res(Shape3(ndev,5,2));
  TensorContainer<cpu, 3, float> tscpu(Shape3(ndev,5,2));
  TensorContainer<cpu, 3, float> rescpu(Shape3(ndev,5,2));
  std::vector<int> devs;
  for (int i = 0; i < ndev; ++i) {
    devs.push_back(i);
    tscpu[i] = 1.0 + i;
  }
  mshadow::Copy(ts, tscpu);
  ps->Init(devs);
  for (int i = 0; i < ndev; ++i) {
    ps->Push(ts[i], 3, i);
    ps->PullWait(3, i);
    ps->PullReq(res[i], 3, i, 0,
                );
  }
  for (int i = 0; i < ndev; ++i) {
    ps->PullWait(3, i);
  }
  mshadow::Copy(rescpu, res);
  for (int i = 0; i < ndev; ++i) {
    printf("----dev=%d----\n", i);
    Print2DTensor(rescpu[i]);
  }
  return 0;
}
