// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Lets another program (the committer cp_commit, the prover cp_prove) build a blk::Configured
// (shapes and round plans; blocks/frame/configure.h) from policy.txt, layout.txt and KL.bin with
// the verifier cp_verify's code, and so reject the files that every stage of cp_verify rejects.
//
// Building: compile verifier/src/cp_verify.cpp with -DCP_VERIFY_AS_A_UNIT, which defines
// cp_with_configured() in place of main(), and with the -DBLK_FIELD, -DDIGEST_WORDS,
// -DSERVING_BUILD and -DVF_SOURCE_DIGEST of verifier/build.sh, and link the object. Compile the
// caller with -DBLK_FIELD=2 and the same BLK_HASH. If the caller defines structs whose names
// cp_verify.cpp also uses (the prover: Seg, Call, ACall), first hide the object's other symbols, as
// verifier_unit() in prover/build.sh does: otherwise the linker merges their member functions.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "frame/configure.h"

// Runs what each of cp_verify's three stages (processes: key, leaf, request) runs before it uses a
// key file (`ivk/key.bin`), a leaf file (`ivk/leaf/<k>.bin`), a request or a proof, up to and
// including blk::configure(), which also tests the conditions for hiding on the configuration
// files alone (blocks/frame/hiding_conditions.h) and whether a key file can be made
// (blk::key_fault()). Then every round planned for a key file, a leaf file and a proof must meet
// the target in policy.txt (line `sec`), proof of work counted, and that target must be at least
// asked_bits (rejection name "security-bits").
//
// Last, it calls then() and returns its result. The Configured lives only during that call; it is
// a plain value: copy it to keep it.
// Rejected files end the process: "NO: <name>" on stdout, sometimes after lines that give the
// reason, and exit code 1. then() is not called.
// Files under which cp_verify ends at best with NOT-A-VERIFICATION do reach then(). So do files
// that only cp_verify's parent process rejects, before the stages: cpPolicyHeld() ends a whole call
// (key file, leaf file and proof) with "NO: policy" if `sT` is not `sh` - 4 in policy.txt, but the
// stages also accept `sT` from `sh` + 4 to `sh` + 10.
// Call it once in a process: parsing policy.txt sets file-scope variables, and a second call, even
// inside then(), ends the process.
// policy_txt, layout_txt: the files' bytes, not paths. KL_words: KL.bin as little-endian 32-bit
// words. asked_bits: an integer from 0 to 1,000,000, or the process ends. `quiet`: omits the
// progress line on stdout.
int cp_with_configured(const std::string &policy_txt, const std::string &layout_txt, const std::vector<uint32_t> &KL_words, double asked_bits, bool quiet,
                       const std::function<int(const blk::Configured &)> &then);
