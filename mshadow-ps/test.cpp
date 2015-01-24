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
  ps::ISharedModel<cpu, float> *ps = ps::CreateSharedModel<cpu, float>("local");
  TensorContainer<cpu, 3, float> ts(Shape3(ndev,5,2));
  TensorContainer<cpu, 3, float> res(Shape3(ndev,5,2));
  std::vector<int> devs;
  for (int i = 0; i < ndev; ++i) {
    devs.push_back(i);
    ts[i] = 1.0 + i;
  }
  ps->Init(devs);
  for (int i = 0; i < ndev; ++i) {
    ps->Push(ts[i], 3, i);
    int a = i;
    ps->PullWait(3, i);
    ps->PullReq(res[i], 3, i, 0, [&](Stream<cpu> *stream) {
        printf("hello i=%d, a=%d,remember during callback, do not take local varaible.. \n", i, a);
        ts += 1.0f;
      }
      );
  }
  for (int i = 0; i < ndev; ++i) {
    ps->PullWait(3, i);
    ps->PullWait(3, i);
    printf("----dev=%d----\n", i);
    Print2DTensor(res[i]);
  }
  return 0;
}
