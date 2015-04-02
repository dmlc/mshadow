#include "./dist_async_sum-inl.h"
namespace PS {
App* App::Create(const string& conf) {
  if (IsServer()) {
    return new mshadow::ps::MShadowServerNode<float>(conf);
  }
  return new App();
}
}  // namespace PS


int main(int argc, char *argv[]) {
  PS::StartSystem(argc, argv);
  int ret = 0;
  if (PS::IsWorker()) ret = Run<mshadow::cpu>(argc, argv);
  PS::StopSystem();
  return ret;
}
