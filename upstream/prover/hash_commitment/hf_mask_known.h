// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Known answers for the two rules (hf_mask8() and hf_salt5() in prover/hash_commitment/hf_mask.cuh)
// that derive a weight table's masks (random values encoded with each row) and its tree leaves'
// salts from the table's making key, a ChaCha20 key. Used by tests and by generator_own_test()
// (prover/hash_commitment/hf_mask_own.cuh), the self-test [own-test-generator] of cp_commit and
// cp_prove. Some rows have large indices (weight-table row r above 255, block g of eight masks
// above 65,535, tree leaf I of 2^32 or more), to catch truncated indices.
// The rows are KNOWN2 (field modulus p = 2^64 - 2^32 + 1) of toy/mask_rules.py, as its option
// --header 2 prints them: do not edit them here. That program was written separately from every
// C++ implementation. But its field-dependent parts and those of
// prover/hash_commitment/hf_mask_ref.h and prover/second_writing/mask_generator.h were written
// together, from the rules' wording: these three are separate from
// prover/hash_commitment/hf_mask.cuh, not from one another. The rows test the two rules. In
// toy/mask_rules.py, block() follows RFC 8439, section 2.3; the rows do not show a fault that it
// shares with a C++ implementation of ChaCha20's block function (sources listed in
// THIRD_PARTY.md, "ChaCha20"). The known answer from RFC 8439, section 2.3.2, tests that function,
// in rfc_vector_holds() of toy/mask_rules.py and in generator_own_test(); no row here comes from
// OpenSSL. A test that is not part of this repository compares chacha20_block() of
// prover/hash_commitment/hc_chacha.cuh with "openssl enc -chacha20" on random keys and inputs.
#pragma once
#if defined(BLK_FIELD) && BLK_FIELD != 2
#error "BLK_FIELD, if it is given, must be 2 (p = 2^64 - 2^32 + 1)"
#endif
#include <cstdint>

// Masks of weight-table row r under test key n: m[sigma] = mask[r][8 * g + sigma]. The test keys
// are public: hc_test_seed_word() in prover/hash_commitment/hc_format.h. `tries`: the number of
// ChaCha20 blocks rejected. k gives the rejection threshold: 8 is the programs' own, the largest
// multiple of the field modulus p that is at most 2^64; 1 to 7 is that many eighths of it, the
// eighth rounded down, to make retries common. Without HC_MASK_LOW_LIMIT=1 hf_mask8() ignores its
// `limit` argument: compare it only with rows of k = 8. Three rows have a retry at k = 8.
struct HfKnownMasks {
  int n;
  uint32_t r, g;
  int k;
  uint32_t tries;
  uint64_t m[8];
};
// The salt of tree leaf I under test key n: five field elements.
struct HfKnownSalt {
  int n;
  uint64_t I;
  uint64_t s[5];
};

constexpr HfKnownMasks HF_KNOWN_MASKS[] = {
    {0, 0u, 0u, 8, 0u, {0x7fb30dce10cc8188ull, 0xde8b7f2bf388be07ull, 0x22486e03f6a91eddull, 0x55dba716074d26e9ull, 0xb3b31db7b2a70c19ull, 0xc729952061d0fa87ull, 0x246d8ef053f50181ull, 0xa6272ca3ba8cd6f6ull}},
    {0, 1u, 0u, 8, 0u, {0xbd9e469cc24680d7ull, 0x3f7de042cba3a832ull, 0x1da6a959784b24bbull, 0x94b8b2bbe840e17aull, 0x323224fba62c289full, 0x1eafb7553af55566ull, 0x9ea25b02e66900bcull, 0xea0877ca0fa9348full}},
    {0, 0u, 1u, 8, 0u, {0xeaf3d8e0acee02e3ull, 0xa3662a5a19bdcb56ull, 0xf91a3f2191486d90ull, 0xdc59d6101717d430ull, 0x3a00221ce0f1076eull, 0x4f1654db02c244f3ull, 0xa09338032cecde52ull, 0x760b69b390730edcull}},
    {0, 262143u, 4194303u, 8, 0u, {0x0abcbeacd144e750ull, 0x5437826087310635ull, 0x79fb19ef442fd9deull, 0xbdea8287e02c14faull, 0x06bc1365158eaf62ull, 0x240062fef2331410ull, 0x6fb961cf4a9b9e20ull, 0x2968ef36ce5cfadbull}},
    {0, 0u, 0u, 7, 0u, {0x7fb30dce10cc8188ull, 0xde8b7f2bf388be07ull, 0x22486e03f6a91eddull, 0x55dba716074d26e9ull, 0xb3b31db7b2a70c19ull, 0xc729952061d0fa87ull, 0x246d8ef053f50181ull, 0xa6272ca3ba8cd6f6ull}},
    {0, 1u, 0u, 7, 3u, {0x0d61561d722fdab2ull, 0x6749d4241fb0d2abull, 0xba151e5ef275a404ull, 0xb308668100c166e5ull, 0x071366dd429aafc6ull, 0xcbe057b0b73cceccull, 0x4a783567b8511331ull, 0x929befb36d00318dull}},
    {0, 2u, 0u, 7, 2u, {0x0cb53b70fd5dfafdull, 0x72084982f9ab51d5ull, 0x8ff62b6a00f7c0abull, 0x641b4b636dd1bf9aull, 0x8f4057cf542cb780ull, 0x49cbe1218ea1070aull, 0x48e93f1493144171ull, 0x67fa2a087ef41e46ull}},
    {0, 3u, 0u, 7, 1u, {0x95e9cb3a8f5f0cd3ull, 0x76d39a200d90c53cull, 0xa789fc8e0f0b6cddull, 0x1498b777c4a2637full, 0x569ff2f8dcbb09dbull, 0x7dce843835bb5572ull, 0x47636dce27e1c069ull, 0x5f2ac92d1b0e7e41ull}},
    {0, 6u, 0u, 6, 3u, {0x914b2c290b5df887ull, 0x71b9e902bd9eb873ull, 0xaf1ef85cacade08bull, 0xab56d1f16da3e6efull, 0x1a53f370c8538a93ull, 0xb770854edbfc6ad0ull, 0x2343fe75ff78e67aull, 0x5654b47ef10db825ull}},
    {255, 0u, 0u, 8, 0u, {0x7c0d53fb99c4a7cbull, 0xa8a0477d1ae578dcull, 0xe1d8a3673be830e1ull, 0x3f668bb656628cc5ull, 0x2741bd7af6076debull, 0xba490e9ca7ceec13ull, 0x346a6296fa07534aull, 0x8f749f8150df3e33ull}},
    {1, 0u, 0u, 8, 0u, {0x7b93ea1f2ddfbe14ull, 0x7c413591cd609bb6ull, 0x59130c1fc945431bull, 0xf6fb6028d67ded5aull, 0x95d52faafcd67aa9ull, 0xe38d3e192adc3317ull, 0x748d6f5712a56ac9ull, 0xeb841877d1c7f74bull}},
    {1, 1u, 0u, 8, 0u, {0xb8fe32c354f6bc66ull, 0xd0e5084635b7cf3bull, 0x8fbf482a4c60053bull, 0x242d7fc67ed0046full, 0xd51159080902bf0eull, 0x1aa684af8970be7cull, 0xc2ef16f732b2933bull, 0x3cb1269ccf8b7514ull}},
    {1, 0u, 1u, 8, 0u, {0x795a1b0a16dbf252ull, 0x66c44772985251ecull, 0xa7191c8da328d842ull, 0x9e849103685df522ull, 0xbffa51baf4179216ull, 0xce79380ed900cf5cull, 0xe39dc2909d602c61ull, 0xcb287ace818fb400ull}},
    {1, 256u, 0u, 8, 0u, {0xf0c1adf4389383eaull, 0x09f67ba2283b7467ull, 0x66ee1b39b13879feull, 0x96d9b5b1a8a260acull, 0xf5ef88e6f3949282ull, 0x762340b66b435b76ull, 0x806d7eb4e9e5dd01ull, 0x00984e3b74acc1b9ull}},
    {1, 0u, 65536u, 8, 0u, {0x4dcee7b17551dc53ull, 0xb6bec09f3a5f1d0eull, 0xc7b4e50cf3633f2eull, 0x9ce45cb0e02ea339ull, 0x79aee156ff5f153aull, 0xa0c6ae1d0eaf34a5ull, 0x8f63514181fcb674ull, 0x7c7d14187367ab29ull}},
    {1, 65536u, 0u, 8, 0u, {0x8e30bdfba5c6ff84ull, 0x6d6bc5891f560fe0ull, 0x8b020ec15e709cd5ull, 0xced9ee13d23860f0ull, 0x82895c580a6b5f8bull, 0xc9e22736a0b11170ull, 0x65d5f0ed442ed21aull, 0xf5f0ddcaab00c1f3ull}},
    {1, 2047u, 5505023u, 8, 0u, {0x31ada3751f23c39full, 0x1b21c1e689eb37a0ull, 0x1721f9efbad82adaull, 0x7da0608b4f4e5b74ull, 0xd8bab20fec2227c8ull, 0x4b0f1a5ada2dee6bull, 0x59b6f3d997f18147ull, 0x74c907247d86f192ull}},
    {1, 8191u, 5505023u, 8, 0u, {0xad8ca6767ff6eb94ull, 0x2f2ff42402b33bb3ull, 0x21b9e630bfcbc9acull, 0x8c4a75ddd053f81aull, 0x723dfbc14b084adeull, 0x01964ec3b13cab03ull, 0x4573a228df42c5ddull, 0x57472bcbc9b0041full}},
    {1, 2047u, 5505022u, 7, 2u, {0xa72824a6af8fd6fbull, 0x4ed7f5849866b4d8ull, 0xd3558cf130b73200ull, 0x9e5e0fce298bddbdull, 0x696cf46f5642f947ull, 0xa293335265e29050ull, 0xc323fef3d1d100ccull, 0xc77ea8062df015e7ull}},
    {1, 2047u, 5505020u, 7, 1u, {0x8d6f84c5c6c8de17ull, 0x0ea1c63c318c8814ull, 0x9c8be78537c3e6deull, 0x4ad0fe133ceb6ff9ull, 0x0e2c440af3bc315aull, 0xc0ec3f70471013e5ull, 0x5b2ca073d43a3ac5ull, 0x0e16b256d13cd65dull}},
    {1, 235u, 3890687u, 8, 1u, {0xe88fa190e8b72544ull, 0xd93feb442c403780ull, 0xd5fbf391c9e619a8ull, 0xb0fe467a7a828c18ull, 0xd64b78ff709e1812ull, 0xc3738c0375c94958ull, 0x376964af723caba4ull, 0x198da622a8cbad32ull}},
    {1, 292u, 5264827u, 8, 1u, {0xcddcfb68434c1081ull, 0xbba4070d5e67d90bull, 0xf288c70b1c4e7f23ull, 0x4924fa59a91259e6ull, 0xef2c39855d42af32ull, 0xdd5fc9f67000bea2ull, 0x836a441e16f67a79ull, 0xbc87f80414391420ull}},
    {1, 391u, 1987782u, 8, 1u, {0x634023c494c9b60dull, 0x332ae5cfafcd67e6ull, 0xde9455fe623215c1ull, 0x267ed6297bfd349cull, 0x92c46a5297dff9cdull, 0xea39516316f4bc09ull, 0xa821fff66fb92f20ull, 0xb47d0a0e0973eadaull}},
};
constexpr HfKnownSalt HF_KNOWN_SALTS[] = {
    {0, 0ull, {0xdbf90a4d1108f198ull, 0x69ba481bb13890bbull, 0x9f5ba572f66a3cedull, 0xb36142df94e8d0d8ull, 0xfe7b3d06bbdee6b6ull}},
    {1, 0ull, {0xdca55308f7250e4full, 0x9092d81a9d4057e9ull, 0xa72c8c7d77a022ccull, 0xbd3053c098f71accull, 0x969f496996f6ad10ull}},
    {1, 1ull, {0x2cfce91e068aaf42ull, 0x4f6d5a1ac3c0087bull, 0x476dd89629266fb1ull, 0xec915950efd00797ull, 0xe57dc2197cb59ed1ull}},
    {1, 4294967295ull, {0xa5caa89f5acbff7aull, 0xe9ebc82dbaf57357ull, 0xa3ee12134b3238abull, 0xe550de77b742d4ecull, 0xdf3d896c35609018ull}},
    {1, 4294967296ull, {0xea595a5156bc953aull, 0x92a051050c88d00dull, 0x087ab2a255705480ull, 0x8a542c27a4431d9bull, 0x947a59b846ce3ecbull}},
    {1, 4294967297ull, {0x6f3267028bb1b2beull, 0x61b85a65aa4cd906ull, 0x8905f0c015ae3df4ull, 0x8b94a9b7075463bbull, 0xec824b880a5693d8ull}},
    {1, 8589934591ull, {0x999a80af389c572full, 0xe3f1262c45c1305aull, 0x5ddca460e6bc79dfull, 0x0cc81c88bdf27e18ull, 0x9e6e1d8215e27f4cull}},
    {255, 8589934591ull, {0xda0d70b8ad579bb1ull, 0xf3281ca189d66ba6ull, 0xe94732a5f64b3a2dull, 0x2e5a8dc71542db41ull, 0x2937a769e0ee674aull}},
};
constexpr int HF_KNOWN_MASKS_ROWS = sizeof HF_KNOWN_MASKS / sizeof HF_KNOWN_MASKS[0];
constexpr int HF_KNOWN_SALTS_ROWS = sizeof HF_KNOWN_SALTS / sizeof HF_KNOWN_SALTS[0];
