#include "./dist_async_sum-inl.h"

#include "gflags/gflags.h"
#include "glog/logging.h"
namespace PS {

App* App::Create(const string& conf) {
  auto my_role = MyNode().role();
  if (my_role == Node::SERVER) {
    return CreateServerNode(conf);
  }
  return new App();
}
App* CreateServerNode(const std::string& conf) {
  return new mshadow::ps::MShadowServerNode<float>(conf);
}
} // namespace PS
// int WorkerNodeMain(int argc, char *argv[]) {



int main(int argc, char *argv[]) {
  auto& sys = PS::Postoffice::instance();
  sys.Run(&argc, &argv);
  int ret = 0;
  if (PS::MyNode().role() == PS::Node::WORKER) {
    ret = Run<mshadow::cpu>(argc, argv);
  }
  sys.Stop();
  return ret;
}
