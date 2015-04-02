#include "./dist_async_sum-inl.h"

namespace PS {
App* CreateServerNode(const std::string& conf) {
  return new mshadow::ps::MShadowServerNode<float>(conf);
}
} // namespace PS


int WorkerNodeMain(int argc, char *argv[]) {
  return Run<mshadow::cpu>(argc, argv);
}
