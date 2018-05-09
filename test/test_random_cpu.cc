#include "test_random.h"

int main() {
  InitTensorEngine<cpu>();
  Stream<cpu> *stream = NewStream<cpu>(false, false);

  CheckSeed<cpu, int16_t>(stream);
  CheckSeed<cpu, uint16_t>(stream);
  CheckSeed<cpu, int32_t>(stream);
  CheckSeed<cpu, uint32_t>(stream);
  CheckSeed<cpu, int64_t>(stream);
  CheckSeed<cpu, uint64_t>(stream);
  CheckSeed<cpu, float>(stream);
  CheckSeed<cpu, double>(stream);
  CheckSeed<cpu, hf_t>(stream);

  CheckSequence<cpu, int16_t>(stream);
  CheckSequence<cpu, uint16_t>(stream);
  CheckSequence<cpu, int32_t>(stream);
  CheckSequence<cpu, uint32_t>(stream);
  CheckSequence<cpu, int64_t>(stream);
  CheckSequence<cpu, uint64_t>(stream);
  CheckSequence<cpu, float>(stream);
  CheckSequence<cpu, double>(stream);
  CheckSequence<cpu, hf_t>(stream);

  CheckGetRandInt<cpu, int16_t>(stream);
  CheckGetRandInt<cpu, uint16_t>(stream);
  CheckGetRandInt<cpu, int32_t>(stream);
  CheckGetRandInt<cpu, uint32_t>(stream);
  CheckGetRandInt<cpu, int64_t>(stream);
  CheckGetRandInt<cpu, uint64_t>(stream);
  CheckGetRandInt<cpu, float>(stream);
  CheckGetRandInt<cpu, double>(stream);
  CheckGetRandInt<cpu, hf_t>(stream);

  CheckSampleUniformInteger<cpu, int16_t>(stream);
  CheckSampleUniformInteger<cpu, uint16_t>(stream);
  CheckSampleUniformInteger<cpu, int32_t>(stream);
  CheckSampleUniformInteger<cpu, uint32_t>(stream);
  CheckSampleUniformInteger<cpu, int64_t>(stream);
  CheckSampleUniformInteger<cpu, uint64_t>(stream);

  CheckSampleUniformReal<cpu, float>(stream);
  CheckSampleUniformReal<cpu, double>(stream);
  CheckSampleUniformReal<cpu, hf_t>(stream);

  CheckSampleGaussian<cpu, float>(stream);
  CheckSampleGaussian<cpu, double>(stream);
  CheckSampleGaussian<cpu, hf_t>(stream);

  DeleteStream(stream);
  ShutdownTensorEngine<cpu>();
  cout << "All checks passed" << endl;
}
