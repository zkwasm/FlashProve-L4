// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The prover program cp_prove: it proves one batch of requests, or serves batch after batch on one
// GPU until stopped (options: prover/prover/arguments.h; exit codes: prover/prover/ends.h). main()
// runs a `Prover` (prover/prover/prover.h). cp_prove supports only parameter files (`policy.txt`)
// whose line `prof` is 20: attention is then proved too. VERSION_WORD (prover/common/setting.h)
// identifies the proof format. Called with -clear, cp_prove instead removes what an earlier run
// staged and never released (Spool::clear(), prover/prover/spool.h).
#include "common/zeroing_heap.h"
#include "prover/inputs.cuh"
#if !(defined(CHAIN_HOST_ENGINE) && CHAIN_HOST_ENGINE)
#error "cp_prove must be built with CHAIN_HOST_ENGINE=1 (./build.sh sets it): common/element_from_hash.h depends on this switch"
#endif
#include "prover/chain_moments.cuh"
#include "prover/layer_setup.cuh"
#include "prover/layer_walk.cuh"
#include "prover/other_setup.cuh"
#include "prover/stream_plan.cuh"
#include "prover/chain_coins.cuh"
#include "prover/chain_frame.cuh"
#include "prover/header.cuh"
#include "prover/layer_prove.cuh"
#include "prover/request_member.cuh"
#include "prover/request.cuh"
#include "prover/in_step.cuh"
#include "prover/wave_order.cuh"
#include "prover/opening_in_process.cuh"
#include "prover/close_memory.cuh"
#include "prover/fold_walk.cuh"
#include "prover/isk.cuh"
#include "prover/service.cuh"

// Re-executes the program with its environment reduced to HOME, TMPDIR, CUDA_VISIBLE_DEVICES and
// CUDA_DEVICE_ORDER (for root also SUDO_UID, which the function then removes), unless `isk.txt`,
// in the secret key directory (`-isk`), states that the weights are synthetic (cp::made_up()) and
// the seed is a test seed. Reason: the private randomness from a randomly generated seed is hidden,
// a model's weights are in GPU memory whatever the seed, and some environment variables make the
// driver write GPU memory to a file. With synthetic weights and a test seed nothing is hidden, and
// tools that enter through the environment (a sanitizer, a profiler) keep working. See
// reduced_env_start() (prover/common/reduced_env_start.h) for what the re-execution does and does not protect
// against.
//
// No option states what the weights and the seed are, so `isk.txt` is read here and again in
// Prover::read_isk(); nothing compares the two reads. Arguments::isk_on_the_line() finds the
// directory by running the real parser. If the command line is faulty or `isk.txt` cannot be read
// or parsed, the run continues and Prover::read_inputs() rejects it. With any other weights or
// seed, a run with a test-only environment variable set (test_word_in_the_environment()) is not
// re-executed either: read_isk() or read_inputs() rejects it. After a re-execution neither would
// see the variable, and the run would continue without a message.
//
// Every production build has the restricted output mode (`closed_front`:
// prover/prover/closed_front.h): ClosedFront::enter() has then re-executed the program, whatever
// `isk.txt` states. Builds without it reject a model's files (`tests-build-on-a-model`,
// Prover::read_isk()).
//
// Call it at the start of main(). Before the call the program must print nothing unless it then
// exits, create no file, generate no random value, call no CUDA function, and leave argv, the
// environment and the open descriptors unchanged. For main() below and any other program that
// includes this file and loads the weights as cp_prove does. Returns 0 to continue, or 2 after
// printing the `no-fresh-start` message.
[[maybe_unused]] static int second_start_unless_nothing_is_hidden(const char *program, int argc, char **argv) {
  const std::string isk = closed_front.on ? std::string() : Arguments::isk_on_the_line(argc, argv); // if on, `closed_front.enter` has already re-executed the program
  cp::IskText text;
  if (!isk.empty() && cp::read_isk_text(cp::isk_txt(isk), text) && !(cp::made_up(text) && text.test_seed) && !test_word_in_the_environment())
    if (const char *why = geteuid() ? reduced_env_start(argv, {CLOSED_KEEPS}) : reduced_env_start(argv, {CLOSED_KEEPS_FOR_ROOT})) {
      fprintf(stderr, "%s: refused (no-fresh-start): the program cannot start a second time with a reduced environment: %s. Nothing was drawn or written\n", program, why);
      return 2;
    }
  // The account that called sudo, or 0, set before the first call of Spool::open_by_parts(), which
  // reads it. For root, SUDO_UID is kept across the re-execution, whether enter() or this function
  // made it. take_who_typed_sudo() (prover/common/walked_path.h) reads the variable once and
  // removes it from the environment: afterwards no library and no child process finds it. This is
  // done here, not in main(), so that every program that calls this function removes it.
  Spool::typed_sudo = take_who_typed_sudo();
  return 0;
}

#ifndef CP_PROVE_NO_MAIN // defined by programs that include this file and define `main`
#ifdef TEST_HOOKS
#error "TEST_HOOKS is only for test programs with a main of their own: never build the prover's own main with it"
#endif
#ifdef LR_TEST_KEYS
#error "never build the prover's own main with LR_TEST_KEYS"
#endif
int main(int argc, char **argv) {
  if (argc == 2 && !strcmp(argv[1], "--setting")) { // print only the compiled setting, then exit
    print_setting(-1);
    return 0;
  }
  if (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) { // print only the usage text, then exit
    Arguments::usage();
    return 0;
  }
  if (argc > 1 && !strcmp(argv[1], "-clear")) { // only remove what an earlier run staged, then exit
    std::string said;
    Spool::typed_sudo = take_who_typed_sudo();
    const int status = Spool::clear(argc == 3 ? argv[2] : "", said);
    fputs(said.c_str(), stdout);
    return status;
  }
  // Keep enter() first after the three exits above: it can re-execute the program (closed_front.h).
  closed_front.enter(argv);
  if (int rc = second_start_unless_nothing_is_hidden("cp_prove", argc, argv)) return rc;
  static Prover P;
  return end_of_main(P.run(argc, argv));
}
#endif
