// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* frame_as_the_verifier_reads compares what the prover's code derives from a commitment's three
   public files with what the verifier's code derives from them. cp_prove holds no verifier code,
   so this test is the only place where the two readings meet. It runs cp_prove's set-up up to
   Prover::chain_make() and calls cp_with_configured() (verifier/src/cp_configured.h).

   Byte for byte it compares the version word and the plan, which begin every chain's first query,
   for a proof, a key file and a leaf file (`entry`), and all further parameters of a proof's first
   chain. A difference there would make proofs fail. By value it compares the range checks on which
   each side tested the hiding conditions. They are in no first query, so no proof would fail if
   they differed, and no other test compares the two lists.

   Usage: frame_as_the_verifier_reads <bits> [-turn-one-bound] -isk <directory> -spool <directory> [options]
   <bits>: the bits asked of the verifier's code, 128 unless the commitment was made for fewer with
   cp_commit's -test-bits. -turn-one-bound, which must be the second word, adds 1 to the bound of
   the verifier's first range check before the comparison. The run must then print that check's
   numbers as differing. That shows that the comparison of the range checks sees this one number
   changed, and nothing more. The other arguments are as in cp_prove's form with -spool. Give -spool a
   new, empty directory: the tool opens it as cp_prove does, so it would act on what a service left
   in a used one. It needs a GPU: the shapes come out of the sub-proofs' set-ups.
   It prints "frame_as_the_verifier_reads: the same" and exits with code 0, or prints what differs
   and exits with code 1. Files that the verifier's code rejects end the process in that call, after "NO: <name>", also
   with exit code 1. It takes made-up weights alone: every build but the production one refuses a
   model's files, and this program cannot be that build. Built by prover/build.sh
   frame_as_the_verifier_reads. */
#define CP_PROVE_NO_MAIN
#ifdef TEST_HOOKS
#error "never build frame_as_the_verifier_reads with TEST_HOOKS"
#endif
#ifdef LR_TEST_KEYS
#error "never build frame_as_the_verifier_reads with LR_TEST_KEYS"
#endif
#include "prover/cp_prove.cu"
#include "cp_configured.h" // ../verifier/src: the verifier's own way from the three files to a blk::Configured

// Prover::run()'s set-up steps up to chain_make(), in the same order (prover/prover/service.cuh).
// Left out: what only prints, closed_front.close(), stop_if_asked().
static int set_up(Prover &P, int argc, char **argv) {
  P.argc = argc;
  P.argv = argv;
  P.makes_no_response = true; // no proof, no opening, no release
  if (int rc = P.read_inputs()) return rc;
  if (int rc = P.the_ranges_are_guarded()) return rc;
  if (int rc = P.open_the_spool()) return rc;
  if (int rc = P.set_up_the_opening_library()) return rc;
  if (int rc = P.test_the_coins()) return rc;
  if (int rc = P.read_the_seed()) return rc;
  if (int rc = P.make_tables()) return rc;
  if (int rc = P.plan_memory()) return rc;
  if (int rc = P.setup_g_forks()) return rc;
  if (P.args.close_on_the_slab > 0)
    fork_memory.single_base =
        std::min(ForkMemory::rounded((size_t)(P.args.close_on_the_slab * 1e9)), fork_memory.arena_first);
  if (int rc = P.setup_attention_forks()) return rc;
  if (int rc = P.setup_head_fork()) return rc;
  if (int rc = P.plan_stream_ids()) return rc;
  if (int rc = P.setup_buffers()) return rc;
  if (int rc = P.build_header()) return rc;
  if (int rc = P.plan_sections()) return rc;
  return P.chain_make();
}

// Returns what differs between two lists of range checks, or "" if nothing does.
static std::string range_checks_differ(const std::vector<blk::RangeCheck> &checks, const std::vector<blk::RangeCheck> &others) {
  static_assert(sizeof(blk::RangeCheck) == 5 * 8 + 2 * 4, "a range check has a new member: compare and print it below");
  auto numbers = [](const blk::RangeCheck &c) {
    return "rows " + std::to_string(c.rows) + ", columns " + std::to_string(c.columns) + ", combinations " + std::to_string(c.combinations) + ", bound " + std::to_string(c.bound) + ", range " +
           std::to_string(c.range) + ", digits " + std::to_string(c.digit_bits[0]) + " + " + std::to_string(c.digit_bits[1]);
  };
  const std::string handed = "the range checks handed to the conditions of hiding: ";
  if (checks.size() != others.size()) return handed + std::to_string(checks.size()) + " here, " + std::to_string(others.size()) + " in the verifier";
  for (size_t i = 0; i < checks.size(); i++)
    if (numbers(checks[i]) != numbers(others[i]))
      return handed + "check " + std::to_string(i) + " of " + std::to_string(checks.size()) + ": " + numbers(checks[i]) + " here; " + numbers(others[i]) + " in the verifier";
  return "";
}

// Returns what differs between the prover's reading and the verifier's, or "" if nothing does.
static std::string what_differs(Prover &P, double bits, bool turn_one_bound) {
  std::string differs;
  auto parameters = [](const blk::Configured &c, const blk::FilePlan &file, size_t chain) {
    blk::Binder b(0, file.chains[chain].plan); // the label is no parameter
    if (&file == &c.proof() && chain == 0) (void)blk::bind_request_parameters(b, c);
    return b.parameter_bytes();
  };
  const blk::Configured &frame = *P.frame;
  const int status = cp_with_configured(P.policy.bytes, P.layout_bytes, P.KL, bits, true, [&](const blk::Configured &theirs) {
    const blk::FilePlan *mine[3] = {&frame.proof(), &frame.key(), &frame.entry()}, *its[3] = {&theirs.proof(), &theirs.key(), &theirs.entry()};
    for (int f = 0; f < 3 && differs.empty(); f++) {
      if (mine[f]->chains.size() != its[f]->chains.size()) differs = mine[f]->name + ": another number of chains";
      for (size_t i = 0; differs.empty() && i < mine[f]->chains.size(); i++)
        if (parameters(frame, *mine[f], i) != parameters(theirs, *its[f], i)) differs = mine[f]->name + ", " + mine[f]->chains[i].name;
    }
    std::vector<blk::RangeCheck> others = theirs.range_checks();
    if (turn_one_bound && !others.empty()) others[0].bound++;
    if (differs.empty()) differs = range_checks_differ(frame.range_checks(), others);
    return 0;
  });
  return status ? "the verifier's code stopped with an error before the comparison" : differs;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("usage: frame_as_the_verifier_reads <bits> [-turn-one-bound] -isk <directory> -spool <directory> [options]\n");
    return 2;
  }
  closed_front.enter(argv);
  // <bits> is no option of cp_prove's, so this call finds no -isk and the program does not start a
  // second time. It runs on made-up weights alone, where the reduced environment guards nothing.
  if (int rc = second_start_unless_nothing_is_hidden("frame_as_the_verifier_reads", argc, argv)) return rc;
  const double bits = Arguments::number(argv[1]);
  if (bits < 0) {
    printf("frame_as_the_verifier_reads: the first argument, <bits>, must be a number of 0 or more, such as 128\n");
    return 2;
  }
  const bool turn_one_bound = argc > 2 && !strcmp(argv[2], "-turn-one-bound");
  const int ours = turn_one_bound ? 2 : 1;
  argv[ours] = argv[0]; // cp_prove's arguments begin behind this program's
  argc -= ours, argv += ours;
  static Prover P;
  if (int rc = set_up(P, argc, argv)) return end_of_main(rc);
  const std::string differs = what_differs(P, bits, turn_one_bound);
  if (differs.empty())
    printf("frame_as_the_verifier_reads: the same\n");
  else
    printf("frame_as_the_verifier_reads: the two readings of the public files differ: %s\n", differs.c_str());
  return end_of_main(differs.empty() ? 0 : 1);
}
