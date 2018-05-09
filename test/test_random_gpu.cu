#include "test_random.h"

template<typename Rng, typename DType>
__global__ void CheckSampleUniformScalar(bool* check) {
  printf("Check SampleUniformScalar <%s>\n", typenameString<DType>());
  Rng rand;
  mshadow::UniformRealDistribution<gpu, Rng, DType> dist;
  const unsigned N = 1000000;
  *check = true;
  double x1 = 0;
  double x2 = 0;
  for (unsigned i = 0; i < N; ++i) {
    double x = dist(rand);
    x1 += x;
    x2 += x * x;
    *check = *check && (0 <= x && x < 1);
  }
  *check = *check && abs(x1 / N - 0.5) < 0.01 && abs(x2 / N - 1. / 3) < 0.01;
}

template<typename Rng, typename DType>
__global__ void CheckSampleGaussianScalar(bool* check) {
  printf("Check SampleGaussianScalar <%s>\n", typenameString<DType>());
  Rng rand;
  mshadow::GaussianDistribution<gpu, Rng, DType> dist;
  const unsigned N = 1000000;
  double x1 = 0;
  double x2 = 0;
  for (unsigned i = 0; i < N; ++i) {
    double x = dist(rand);
    x1 += x;
    x2 += x * x;
  }
  *check = abs(x1 / N) < 0.01 && abs(x2 / N - 1) < 0.01;
}

template<typename DType>
void CheckSampleScalar(Stream<gpu>* stream) {
  cudaStream_t s = Stream<gpu>::GetStream(stream);
  bool* check_dev;
  cudaMalloc(&check_dev, sizeof(check_dev));
  bool check = false;
  using RandomBitGenerator = typename std::conditional<sizeof(DType) <= 4, PCGRandom32, PCGRandom64>::type;

  CheckSampleUniformScalar<RandomBitGenerator, DType><<<1, 1, 0, s>>>(check_dev);
  cudaMemcpy(&check, check_dev, sizeof(bool), cudaMemcpyDeviceToHost);
  assert(check);

  CheckSampleGaussianScalar<RandomBitGenerator, DType><<<1, 1, 0, s>>>(check_dev);
  cudaMemcpy(&check, check_dev, sizeof(bool), cudaMemcpyDeviceToHost);
  assert(check);
}

int main() {
  InitTensorEngine<gpu>();
  Stream<gpu> *stream = NewStream<gpu>(false, false);

  CheckSeed<gpu, int16_t>(stream);
  CheckSeed<gpu, uint16_t>(stream);
  CheckSeed<gpu, int32_t>(stream);
  CheckSeed<gpu, uint32_t>(stream);
  CheckSeed<gpu, int64_t>(stream);
  CheckSeed<gpu, uint64_t>(stream);
  CheckSeed<gpu, float>(stream);
  CheckSeed<gpu, double>(stream);
  CheckSeed<gpu, hf_t>(stream);

  CheckSequence<gpu, int16_t>(stream);
  CheckSequence<gpu, uint16_t>(stream);
  CheckSequence<gpu, int32_t>(stream);
  CheckSequence<gpu, uint32_t>(stream);
  CheckSequence<gpu, int64_t>(stream);
  CheckSequence<gpu, uint64_t>(stream);
  CheckSequence<gpu, float>(stream);
  CheckSequence<gpu, double>(stream);
  CheckSequence<gpu, hf_t>(stream);

  CheckGetRandInt<gpu, int16_t>(stream);
  CheckGetRandInt<gpu, uint16_t>(stream);
  CheckGetRandInt<gpu, int32_t>(stream);
  CheckGetRandInt<gpu, uint32_t>(stream);
  CheckGetRandInt<gpu, int64_t>(stream);
  CheckGetRandInt<gpu, uint64_t>(stream);
  CheckGetRandInt<gpu, float>(stream);
  CheckGetRandInt<gpu, double>(stream);
  CheckGetRandInt<gpu, hf_t>(stream);

  CheckSampleUniformInteger<gpu, int16_t>(stream);
  CheckSampleUniformInteger<gpu, uint16_t>(stream);
  CheckSampleUniformInteger<gpu, int32_t>(stream);
  CheckSampleUniformInteger<gpu, uint32_t>(stream);
  CheckSampleUniformInteger<gpu, int64_t>(stream);
  CheckSampleUniformInteger<gpu, uint64_t>(stream);

  CheckSampleUniformReal<gpu, float>(stream);
  CheckSampleUniformReal<gpu, double>(stream);
  CheckSampleUniformReal<gpu, hf_t>(stream);

  CheckSampleGaussian<gpu, float>(stream);
  CheckSampleGaussian<gpu, double>(stream);
  CheckSampleGaussian<gpu, hf_t>(stream);

  CheckSampleScalar<float>(stream);
  CheckSampleScalar<double>(stream);
  CheckSampleScalar<hf_t>(stream);

  DeleteStream(stream);
  ShutdownTensorEngine<gpu>();
  cout << "All checks passed" << endl;
}