#include "common/setting.h"
#include "weights_commitment/common.cuh"
#include "weights_commitment/arguments.h"
#include "weights_commitment/worst_case.cuh"
int main() {
  rf::worst_case::Check check;
  check.sh.d = check.sh.dq = check.sh.f = check.sh.fe = check.sh.V = 256;
  check.sh.dkv = 128; check.sh.dh = 32; check.sh.NE = 2;
  check.begin();
  short *w; CK(cudaMalloc(&w, 32 * 32 * sizeof(short))); CK(cudaMemset(w, 0, 32 * 32 * sizeof(short)));
  const double zero = check.operator_norm(w, 32, 32);
  printf("zero bound %.17g; reason %s\n", zero, check.why_no_bound ? check.why_no_bound : "none");
  if (zero != 0 || check.why_no_bound) {
    printf("zero norm refused: bound %.17g, %s\n", zero, check.why_no_bound ? check.why_no_bound : "no reason");
    check.end(); cudaFree(w); return 1;
  }
  std::vector<short> values(32 * 32, 0);
  for (int i = 0; i < 32; i++) values[i * 32 + i] = 1;
  CK(cudaMemcpy(w, values.data(), values.size() * sizeof(short), cudaMemcpyHostToDevice));
  const double identity = check.operator_norm(w, 32, 32);
  printf("identity bound %.17g; reason %s\n", identity, check.why_no_bound ? check.why_no_bound : "none");
  if (identity < 1 || identity > 6 || check.why_no_bound) return 2;
  CK(cudaMemset(check.scale, 0, 32 * sizeof(double)));
  const double scaled_zero = check.operator_norm(w, 32, 32, check.scale);
  if (scaled_zero != 0 || check.why_no_bound) return 3;
  check.end(); CK(cudaFree(w));
  printf("operator norm: zero=0, identity bound %.17g >= 1, zero row scales=0\n", identity);
}
