// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The service of the prover program cp_prove under one commitment: set-up (run()), the loop over
// batches (serve()) and a batch's steps. With -in, -out it serves one batch. A batch, in order:
//   take      the oldest requests found in the spool directory (-spool) get batch slots (spool.h);
//   start     (`head`) 32 fresh bytes, the commitment's seed and the requests give the batch's leaf
//             table and process key, the key of all its private randomness (`coins`) (isk.cuh);
//   passes    a batch pass (`wave`) proves up to -in-step slots on one read of the weights
//             (in_step.cuh), each in its first chain (a chain: a Fiat-Shamir transcript);
//   opening   the test on the leaf table (batch_opening.h): the batch's shared chain, or shared
//             opening, reads all the weights once per walk (fold_walk.cuh); then the second chains;
//   outcomes  a request with no failure flag (slot.h), in a batch with no cause of failure
//             (ends.h), has its files staged, every other "failed"; the slots are cleared;
//   release   after the next take (service_loop.h) all staged responses get their final names.
// A leaf table (`leaf`) is a further table of the same weights, with its own masks. With -overlap
// batches overlap in time (prove_the_batch()). A batch's log lines and its bytes in its leaf
// table's column count (count.bin) outlive it; no later batch inherits its key or leaf table. Every
// live slot takes all steps, whatever its flags. A request's failure ends nothing, but can change
// things for the batch's other requests, such as bytes of their proofs, and so the shared and
// second chains' challenges and failures that depend on these; host time and disk traffic, and so
// the time of release and, on a nearly full disk, whether a write fails; after a range violation's
// substitute values, its own routing, and so GPU time (request.cuh). Only a test build prints the
// cause; no exit code shows it. But a failed self-check on secret random values ends the service by
// the next end of a batch. A fault of the machine or user (a failed CUDA call, no memory, SIGTERM,
// an unwritable file) ends the process (stop()); the next answers "failed" to what it had taken.
// `room`: GPU buffer of weights; `library`: keeper unit (owns a leaf table); `fork`: sub-proof.
// In identifiers a seat is a batch slot, an answer a response, a member a request, and the fold the shared opening.
#pragma once
#include "common/cp_files.h"
#include "common/refusal_test.h"
#include "prover/closed_front.h"
#include "prover/isk.cuh"
#include "prover/prover.h"
#include "prover/service_loop.h"
#include <thread>

// Prints a run's first lines: what is visible beyond the proofs. Only options and build alter them.
inline bool Prover::name_what_shows() {
  if (!closed_front.say_closed("p258 at most %d requests share one opening, each with its own proof file\n", args.batch))
    return false;
  if (!closed_front.say_closed("p258 timing not hidden: nothing is done to hide the time of a release, so a batch's time shows; "
                               "empty seats are not proved, so the number of requests in a batch shows;%s\n",
                               args.overlap_two ? " with -overlap-two the time of a batch's release also shows whether the "
                                                  "next two batches stood ready, and how long passes of theirs took;"
                               : args.overlap   ? " with -overlap the time of a batch's release also shows whether a next "
                                                  "batch stood ready, and how long that batch's first passes took;"
                                                : ""))
    return false;
#if BUILT_FOR_TESTS_ALONE // 1 except in a production build (prover/common/setting.h)
  if (!closed_front.say_closed("p258 THIS PROGRAM WAS " BUILT_SAYS ": what it makes is for tests alone\n")) return false;
#endif
#ifndef SERVING_BUILD
  if (!args.test_coins.empty() &&
      !closed_front.say_closed("p258 THE BATCHES' COINS COME FROM A TEST'S FILE: nothing is hidden from whoever has it\n"))
    return false;
#endif
  if (!args.public_weights.empty() && !closed_front.say_closed("p258 FOR PUBLIC WEIGHTS: the commitment's seed is public, so nothing is hidden\n")) return false;
  if (!args.measure) return true;
  return closed_front.say_closed(
      "p258 measuring: the lines that begin with measure: hold each batch's times and its number of requests;\n");
}

// Fills the batch's first slots. `taken` must hold 1 to -batch requests; this is not checked.
inline void Prover::seat_the_batch(std::vector<Spool::Request> &taken) {
  batch.B = (int)taken.size();
  for (int j = 0; j < batch.B; j++) {
    Slot &s = batch.slot[j];
    if (taken[j].tokens.empty()) continue; // -in: a malformed file; "failed" is staged already
    s.state = SlotState::Live;
    snprintf(s.id, sizeof s.id, "%s", taken[j].id.c_str());
    s.tokens.swap(taken[j].tokens);
  }
}

// Proves the batch in the slots and begins its opening. Unless the batch becomes a rider,
// it also runs the opening's walks and ends the batch (end_the_batch()). Fills `times`,
// except its start. Returns 0, or a status for a failure of the machine or the program.
//
// A batch that is put aside (-overlap: a rider). read_isk() rejects -overlap on a model's files.
// When a batch's passes have ended and its shared chain has begun (fold_begin()), the next take is
// made. If it takes a request, this function returns: the batch's slots move to its opening's
// `rode`, and a later call ends the rider. Each walk of its opening takes the weights from a batch
// pass of a later batch (the_rider_folds(), the_riders_walk_ends()); walks that find none run alone
// (the_riders_walk_alone()). Every batch has its own Opening from its start: a keeper unit, a GPU
// engine, a randomness generator, a BatchOpening and a thread. There is one more Opening than there
// can be riders (setup_opening()), so a new batch starts in its own while the rider's serves the
// rider, under the rider's leaf table. From that start on the process-wide key (hz::key32_renew())
// is the new batch's. The rest of the rider's shared chain and its second chains draw not under
// that key but from the generators that moved with its slots (swap_the_slots()). All of them are
// listed by prover/tests/coin_contexts_listed.py, "services", and compared with a plain service's.
//
// Two riders (-overlap-two). An opening has two walks, so with one rider and batches of one batch
// pass each, one walk of every opening still runs alone. With three Openings two batches can wait:
// the batch pass of batch b carries the second walk of batch b - 2 and the first of batch b - 1.
// Before that pass's last round batch b - 2 is ended and released, which frees its Opening for
// batch b + 1, and batch b - 1 goes on to its second walk. In addition:
//   - riders are served and ended oldest first; but for each Opening's own `slab` (a buffer kept
//     for a whole opening), memory is lent to one keeper unit at a time: turn_to() exits otherwise;
//   - Spool::release() names all that is staged, so a rider stages nothing, not even the first
//     parts of its proof files, until the rider before it has been released (the_rider_ends());
//   - a rider's writer thread is joined before the next take, so the two never share the spool.
inline int Prover::prove_the_batch(BatchTimes &times) {
  // Joins the writer threads (stage_the_proofs()) on every return unless the batch becomes a rider.
  struct Joins {
    Prover &prover;
    bool at_the_return = true;
    ~Joins() {
      if (at_the_return) prover.join_the_writers();
    }
  } joins{*this};
  if (int rc = test_fault("group-start")) return rc;
  // The requests and their order are fixed (seat_the_batch()): now the batch's key and leaf table.
  const double head_begins = epoch();
  if (const End why = begin_the_batch_on_a_leaf(); why != End::None) {
    if (int rc = the_riders_walk_alone(true)) return rc; // the riders are ended first
    return the_batch_gets_no_proof(why, times);
  }
  times.leaf = batch.leaf;
  times.head = epoch() - head_begins;
  const double waves_begin = epoch();
  for (int first = 0; first < batch.B; first += args.in_step) {
    stop_if_asked();
    const Wave wave{&batch, first, std::min(args.in_step, batch.B - first)};
    // Without -overlap a batch's last batch pass skips loads for a further read of the weights
    // (no_walk_follows says which): the walks queue their own. With -overlap a batch pass or a walk
    // follows at once and uses them, unless this batch pass carries riders: it lends the buffers.
    // Public sources: see weight_stream.cuh, no_walk_follows and room().
    // Each batch pass carries each rider's next walk.
    wave_carries = !riders.empty();
    weight_stream.no_walk_follows = args.overlap ? wave_carries : first + wave.n == batch.B;
    weight_stream.rooms_will_be_lent = wave_carries;
    if (int rc = prove_wave(*this, wave)) return rc;
  }
  wave_carries = false;
  hz::key32_wipe(); // after the last batch pass: wipes the process-wide copy of the process key
  slot_pads_second.wipe_the_key();
  times.waves = epoch() - waves_begin;
  // A rider whose Opening the next batch will need, and whose walks did not all
  // find a batch pass because this batch is short, runs its remaining walks now.
  if (int rc = the_riders_walk_alone(false)) return rc;

  Opening &mine = *opening_of_the_batch;
  turn_to(mine);
  Ending &ending = mine.ending;
  ending.b = &batch;
  ending.times = times;
  ending.shared = End::None;
  if (!args.overlap) stage_the_proofs(mine);
  double mark = epoch();
  auto lap = [&](double &seconds) {
    const double t = epoch();
    seconds += t - mark, mark = t;
  };
  if (int rc = test_fault("opening")) return rc; // a stub of a test hook: it returns 0
  if (int rc = fold_begin()) return rc; // it fills in the time fold_begin
  mark = epoch();
  if (args.overlap) {
    // The take comes after a rider's writer thread has ended and before this batch's
    // starts, so that no writer thread works in the spool directory during a take.
    join_the_writers();
    if (take_ahead && take_ahead() >= 1) {
      swap_the_slots(batch, mine.rode);
      ending.b = &mine.rode;
      ending.shared = shared_with_the_batch, shared_with_the_batch = End::None;
      if (riders.empty()) stage_the_proofs(mine); // otherwise when the rider before it has been released
      mark = epoch();
      take_back_what_is_lent(); // the walk room: the next batch's passes use that scratch memory
      lap(ending.times.fold_walk);
      mine.walks = 0;
      riders.push_back(&mine);
      opening_of_the_batch = nullptr;
      joins.at_the_return = false;
      times = ending.times;
      return 0;
    }
    // No next batch is ready: a rider that still waits (-overlap-two) is ended first.
    if (!riders.empty()) {
      take_back_what_is_lent();
      if (int rc = the_riders_walk_alone(true)) return rc;
      turn_to(mine);
      CK(cudaDeviceSynchronize()); // as in lend_the_slab()
      if (int rc = lend_the_walk_room()) return rc;
    }
    stage_the_proofs(mine);
    mark = epoch();
  }
  if (int rc = fold_walks()) return rc; // it fills in the time fold_next
  lap(ending.times.fold_walk);
  ending.times.fold_walk -= ending.times.fold_next;
  const int rc = end_the_batch(false);
  times = ending.times;
  return rc;
}

// Answers "failed" to every request of a batch that begin_the_batch_on_a_leaf() rejects: its leaf
// table cannot be used, or its 32 fresh bytes could not be stored for -keep-for-the-audit. The
// service then ends (isk_fault). No second table is chosen: the choice would depend on which tables
// can be used. `batch.coins`, the batch's randomness generator, if made, stays and keeps its key.
inline int Prover::the_batch_gets_no_proof(End why, BatchTimes &times) {
  shared_with_the_batch = End::None;
  explicit_bzero(&batch.key_word, sizeof batch.key_word);
  times.cause = why;
  times.leaf = batch.leaf;
  for (int j = 0; j < batch.B; j++) {
    if (batch.slot[j].state == SlotState::Empty) continue;
    if (!spool.stage_failed(batch.slot[j].id)) {
      printf("the spool: %s\n", spool.why);
      stop(End::WriteFailed);
    }
    clear_slot(batch.slot[j]);
  }
  times.staged = epoch();
  return 0;
}

// Starts the writer thread, which stages each request's next token (file "y") and its proof file's
// header (`head`) and first chain while the leaf table is opened. Nothing is written for a request
// with a flag set or, once a self-check on secret random values has failed, for any. A slot's
// failure word is now final (slot.h, flags_host), so a request can still fail only with its whole
// batch; its staged parts are then overwritten with zeros and unlinked (Spool::stage_failed()).
inline void Prover::stage_the_proofs(Opening &o) {
  the_count_is_lasting(o); // no proof or next token is staged before the column count is on disk
  Ending &ending = o.ending;
  const Batch &b = *ending.b;
  uint64_t failed = self_check_failed ? ~(uint64_t)0 : never_released;
  for (int j = 0; j < b.B; j++) failed |= (uint64_t)(flags_of(b.slot[j]) != 0) << j;
  ending.proofs_written = true;
  blk::Writer head; // tag, version word, the batch's leaf table's number (blocks/frame/heads.h)
  blk::write_head(head, blk::TAG_PROOF, b.leaf);
  ending.writer.thread = std::thread([this, &ending, &b, failed, head = head.bytes()] {
    for (int j = 0; j < b.B; j++) {
      const Slot &s = b.slot[j];
      if (failed >> j & 1 || s.state == SlotState::Empty) continue;
      if (!s.part.first) { // a live slot without a flag has run its first chain
        printf("internal error: slot %d has no flag set and no first chain\n", j);
        name_the_end("no-first-chain");
        exit(4);
      }
      ending.proofs_written = spool.stage_answer(s.id, {{"y", &s.y, 4},
                                                        {"proof", head.data(), head.size()},
                                                        {"proof", s.part.first->data(), s.part.first->size()}}) &&
                              ending.proofs_written;
    }
  });
}

// Waits for every writer thread that is at work.
inline void Prover::join_the_writers() {
  for (Opening &o : opening) o.ending.writer.join();
}

// With riders: passes layer group g's units (a unit: one kind of weight of one layer) to each
// rider's keeper unit, for its shared chain. In round g of a batch pass each request has a turn on
// group g (wave_order.cuh). prove_wave() calls this after turn folds_behind() of round g, not the
// round's last: the first turn has waited for every load of the group, and no buffer is released
// before the last. Some units are decoded into the sub-proofs' scratch memory, whose top is lent
// as the keeper unit's workspace (walk_room(), the walk room), to one rider at a time. That memory
// is unused between two turns and zeroed or restored before the next (in_step.cuh), given at least
// two state buffers, which -overlap requires. the_rounds_only_turn: the batch pass has one request,
// whose turn did not count as the round's last, so the routed experts' buffers were not released;
// as their last reader (turn_passed_by()) this function releases them and queues the next loads.
inline int Prover::the_rider_folds(int g, bool the_rounds_only_turn) {
  if (!wave_carries) return 0; // no rider
  stop_if_asked();
  double began = epoch();
  weight_stream.ready(g);
  for (int e = weight_stream.experts_of(g).first; e < weight_stream.experts_of(g).second; e++)
    weight_stream.experts_ready(e);
  list_the_units();
  int rc = 0;
  for (size_t i = 0; i < riders.size() && !rc; i++) {
    turn_to(*riders[i]);
    rc = lend_the_walk_room();
    if (!rc) rc = fold_group_for_opening(g, unit_kind, unit_layer, -2);
    if (!rc) take_back_what_is_lent();
    if (!rc && the_rounds_only_turn && i + 1 == riders.size()) {
      last_of_wave = true;
      weight_stream.turn_passed_by(g);
    }
    const double now = epoch();
    riders[i]->ending.times.fold_walk += now - began, began = now;
  }
  return rc;
}

// With riders: ends the walks that a batch pass has carried. prove_wave() calls it after the
// last layer group, before the first turn of the last round. For each rider, the oldest
// first, it passes the embedding and the output head, which stay on the GPU (-memory tensor-rooms);
// lends the weights' other GPU buffers, which hold nothing that the last round needs, and the
// sub-proofs' scratch memory; and ends the walk. After its last walk the rider is ended. Otherwise
// its keeper unit (`closer`) goes on to the next walk, which the next batch pass carries, and all
// that was lent is taken back. The buffers' next loads are queued once, after all riders.
inline int Prover::the_riders_walk_ends() {
  wave_carries = false;
  const std::vector<Opening *> carried = riders; // `the_rider_ends` removes the oldest of `riders`
  for (Opening *rider : carried) {
    turn_to(*rider);
    Ending &ending = rider->ending;
    const double began = epoch();
    int rc = lend_the_walk_room();
    for (int which = 0; which < 2 && !rc; which++) rc = fold_end_for_opening(which);
    if (!rc) rc = fold_rest(true);
    ending.times.fold_walk += epoch() - began;
    if (rc) return rc;
    int walks;
    {
      InsideTheCloseLibrary inside;
      walks = closer->walks();
    }
    if (++rider->walks == walks) {
      if (int status = the_rider_ends(true)) return status;
      continue;
    }
    const double between = epoch();
    bool more;
    {
      InsideTheCloseLibrary inside;
      more = closer->next_walk(); // waits until the rider's thread stands in its next walk
    }
    take_back_what_is_lent(); // it waits for the GPU
    ending.times.fold_next += epoch() - between;
    if (!more) {
      printf("internal error: a rider's shared chain has ended before the last of its walks over the weights\n");
      return 4;
    }
  }
  if (args.stream_weights) weight_stream.restart(); // the lent buffers' next loads are queued here
  return 0;
}

// With riders, outside a batch pass: runs the oldest rider's remaining walks alone, as without
// -overlap, and ends that rider; then the next rider's. `all`: until no rider is left; else only
// until the current batch can become a rider and the batch after it still finds a free Opening.
inline int Prover::the_riders_walk_alone(bool all) {
  while (!riders.empty() && (all || riders.size() + 2 > (size_t)openings)) {
    Opening &rider = *riders.front();
    turn_to(rider);
    Ending &ending = rider.ending;
    int walks;
    {
      InsideTheCloseLibrary inside;
      walks = closer->walks();
    }
    while (rider.walks < walks) {
      const double began = epoch(), next_before = ending.times.fold_next;
      if (int rc = lend_the_walk_room()) return rc;
      if (int rc = fold_walk()) return rc;
      if (++rider.walks < walks) {
        if (int rc = fold_next()) return rc; // it lends the walk room anew
        take_back_what_is_lent();
      }
      ending.times.fold_walk += epoch() - began - (ending.times.fold_next - next_before);
    }
    if (int rc = the_rider_ends(false)) return rc;
  }
  return 0;
}

// Ends the oldest rider, whose units have all been passed (in_a_pass: see end_the_batch()),
// then has the service loop release its responses and print its log lines (release_now).
// Nothing of a younger batch may be staged at that point, except "failed" for requests that
// got no slot, because Spool::release() gives everything that is staged its final name.
// Last, the next rider's writer thread starts (stage_the_proofs()).
inline int Prover::the_rider_ends(bool in_a_pass) {
  if (riders.empty()) return 0;
  Opening &rider = *riders.front();
  turn_to(rider);
  if (int rc = end_the_batch(in_a_pass)) return rc; // it joins the rider's writer thread
  riders.erase(riders.begin());
  rider_proved = Proved{true, rider.rode.B, rider.ending.times};
  SERVING_OR(, printf("# overlap: the opening of batch %llu rode on the next batch\n", (unsigned long long)batch.number));
  release_now();
  if (!riders.empty()) stage_the_proofs(*riders.front());
  return 0;
}

// Ends the batch of the current Opening (`at_hand`), once its keeper unit has every unit of the
// weights: completes the opening, stages every response, calls batch_end() and clears the slots.
// in_a_pass: the batch is a rider, and the call comes between two turns of a later batch's pass
// (close_memory.cuh lists what is lent then); the caller restarts the weight stream.
inline int Prover::end_the_batch(bool in_a_pass) {
  Ending &ending = at_hand->ending;
  Batch &b = *ending.b;
  BatchTimes &times = ending.times;
  JoinedThread &writer = ending.writer;
  unsigned fold_flags = 0;
  {
    double mark = epoch();
    auto lap = [&](double &seconds) {
      const double t = epoch();
      seconds += t - mark, mark = t;
    };
    // Waits for the shared chain's thread; then runs each request's second chain, in slot order.
    if (int rc = fold_over()) return rc;
    lap(times.fold_finish);
    for (int j = 0; j < b.B; j++)
      if (b.slot[j].state != SlotState::Empty) at_hand->order->second(b.slot[j].part);
    CK(cudaDeviceSynchronize());
    // [own-test-engine-coins] Checks a sample of the shared and second chains' private randomness
    // before any byte of them leaves the process; finish_request() did so for the first chains.
    for (int j = 0; j < b.B; j++)
      if (b.slot[j].state != SlotState::Empty) coins_hold(b.slot[j].coins.get(), "a request's");
    coins_hold(b.coins.get(), "the batch's");
    lap(times.claims);
    InsideTheCloseLibrary inside;
    fold_flags = closer->take_flags();
    closer->loader_seconds(&times.loader_read, &times.loader_rebuild, &times.loader_waited);
    if (const char *stop = closer->keeps_fault()) isk_fault = stop;
    if (const char *name = closer->own_test_fault()) a_self_check_failed(name);
  }
  double behind = epoch();
  auto lap_behind = [&](double &seconds) {
    const double t = epoch();
    seconds += t - behind, behind = t;
  };
  writer.join();
  lap_behind(times.join);
  explicit_bzero(&b.key_word, sizeof b.key_word);
  if (!ending.proofs_written) {
    printf("the spool: %s\n", spool.why);
    stop(End::WriteFailed);
  }
  // Nothing of the batch is released, and no byte of the opening is staged, if this read finds a
  // rejected GPU launch. A kernel launch written with <<<...>>> returns no status: a rejection is
  // kept only in the launching thread's error variable. A later error overwrites it; cudaGetLastError()
  // clears it (NVIDIA, CUDA C++ Programming Guide, "Error Checking"); so did one other call (cuda-status.md).
  // For errors before a kernel runs, the guide wants a read directly after the launch; this one, the serving thread's,
  // comes after its launches for the batch's proofs, also inside the keeper unit, which must use the same CUDA runtime
  // (-cudart shared in prover/build.sh). The opening's thread reads its variable (prover/hash_commitment/hf_keeper.cu,
  // opening_is_over()). The writer thread has been joined, so an exit here meets no other thread of the batch.
  // See also: NVIDIA, CUDA Programming Guide, "Intro to CUDA C++", "Error Checking in CUDA", "Error State",
  // https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html.
  CK(cudaGetLastError());

  // A SELF-CHECK ON SECRET RANDOM VALUES HAS FAILED, in a batch in progress. The service ends here,
  // after the batch's last computing step and before more of it is staged. A proof made with a lost
  // random value still verifies, and another value may be lost where nothing checks. So no batch
  // whose naming has not begun is named: stop() overwrites and removes what is staged, as far as
  // its wipe gets (see open_the_spool()).
  if (self_check_failed) {
    name_the_end(self_check_failed);
    stop(End::SelfCheck);
  }

  // The outcomes. A flag that belongs to no single request, and a flag of the shared opening, count for every slot.
  const unsigned nobodys = take_live_flags(); // read and cleared: the next batch starts clean
  if (nobodys) fail_the_batch(cause_of(nobodys));
  // A rider ends while a later batch is being proved, so a flag that belongs to no request, or
  // a cause raised with fail_the_batch() (a table of constants that a batch pass has changed,
  // say), may come from any batch in progress. It counts for the rider and is recorded below
  // for every other one (Ending::shared, shared_with_the_batch), but not for a batch that
  // starts later. A flag of the rider's own shared opening counts only for the rider.
  const End of_both = &b == &at_hand->rode ? take_the_batchs_cause() : End::None;
  if (of_both != End::None) fail_the_batch(of_both);
  if (ending.shared != End::None) fail_the_batch(ending.shared);
  ending.shared = End::None;
  if (&b == &batch) { // it ends without having been a rider: what older riders' ends have noted for it is still there
    if (shared_with_the_batch != End::None) fail_the_batch(shared_with_the_batch);
    shared_with_the_batch = End::None;
  }
  if (fold_flags) fail_the_batch(cause_of(fold_flags));
  times.cause = take_the_batchs_cause();
  for (int j = 0; j < b.B; j++) {
    const Slot &s = b.slot[j];
    if (s.state == SlotState::Empty) continue;
    End why = times.cause != End::None ? times.cause : cause_of(flags_of(s));
    if (never_released >> j & 1) {
      printf("test mode: seat %d is never released; its flags: %u\n", j, flags_of(s));
      if (why == End::None) why = End::Stopped;
    }
    // SERVING_OR(a, b): `a` in a production build, else `b`; only a test build prints the cause.
    if (SERVING_OR(false, why != End::None))
      printf("# seat %d, request %s: failed: cause %d, flags %08x\n", j, s.id, (int)why, flags_of(s));
    bool staged;
    if (why != End::None)
      staged = spool.stage_failed(s.id);
    else
      // After stage_the_proofs()'s parts: binding tree place and path | shared | second chain.
      staged = spool.stage_answer(s.id, {{"proof", s.part.path.data(), s.part.path.size()},
                                         {"proof", at_hand->order->shared().data(), at_hand->order->shared().size()},
                                         {"proof", s.part.second.data(), s.part.second.size()}});
    if (!staged) {
      printf("the spool: %s\n", spool.why);
      stop(End::WriteFailed);
    }
  }
  lap_behind(times.stage);
  for (int j = 0; j < b.B; j++) // the batch's longest proof: lengths vary with the drawn positions
    if (const Slot &s = b.slot[j]; s.state != SlotState::Empty && s.part.first)
      times.member_bytes = std::max<size_t>(times.member_bytes, cp::PROOF_HEAD_BYTES + s.part.first->size() + s.part.path.size() +
                                                                    at_hand->order->shared().size() + s.part.second.size());
  if (of_both != End::None) { // for the batches that are still in progress
    for (Opening *other : riders)
      if (other != at_hand && other->ending.shared == End::None) other->ending.shared = of_both;
    if (shared_with_the_batch == End::None) shared_with_the_batch = of_both;
  }

  // Ends the batch in its keeper unit, which zeroes its copies of the leaf table's mask seed
  // (batch_end()), and clears the slots (see clear_slot()). The spool keeps the ids and the
  // outcomes for the release.
  if (int rc = batch_end()) return rc;
  for (int j = 0; j < b.B; j++) clear_slot(b.slot[j]);
  b.coins.reset();          // after the order's object (batch_end()), which drew from them
  at_hand->engine.reset();  // after the slots' parts, which held tables and trees on it
  // The opening has used the weights' GPU buffers. Their next loads are queued here or,
  // inside a batch pass, once every rider's walk has ended (the_riders_walk_ends()).
  if (args.stream_weights && !in_a_pass) weight_stream.restart();
  if (memory_grew()) say_the_alarm(End::MemoryGrew);
  if (memory_grew() && times.cause == End::None) times.cause = End::MemoryGrew; // an alarm: no request loses its files
  if (times.cause == End::None) times.cause = ending.alarm; // what the loop noted while the batch was a rider
  ending.alarm = End::None;
  lap_behind(times.end);
  times.file_write = spool.writing_us.exchange(0) / 1e6, times.file_sync = spool.syncing_us.exchange(0) / 1e6;
  times.file_make = spool.making_us.exchange(0) / 1e6, times.file_lock = spool.locking_us.exchange(0) / 1e6;
  times.staged = epoch();
  return 0;
}

// Prints an alarm's constant line, once per process, when its condition first holds, whatever
// becomes of the batch: a line printed only for a batch with no other cause would show which
// batches failed. cut-short depends only on the number of files in in/, memory-grew on how full a
// batch pass is; that memory-grew does not follow the routing to experts has not been shown
// (docs/what-is-not-hidden.md; prover/docs/ends-in-short.md, "The three alarms").
// unreadable-request depends on whether the system lets cp_prove open files in in/ (as a rule: on
// their owners and modes); no byte of them is read for it. It has no number and no file's name.
inline void Prover::say_the_alarm(End alarm) {
  bool &said = alarm == End::MemoryGrew ? said_memory_grew : alarm == End::CutShort ? said_cut_short : said_unreadable;
  if (said) return;
  said = true;
  if (!closed_front.say_closed(alarm == End::MemoryGrew ? "p258 alarm memory-grew: the prover's own GPU memory has risen above its peak at the end of the first pass; the service goes on (prover/docs/ends-in-short.md)\n"
                               : alarm == End::CutShort ? "p258 alarm cut-short: in/ holds more requests than one scan reads, so those that are served need not be the oldest; the service goes on (prover/docs/ends-in-short.md)\n"
                                                        : "p258 alarm unreadable-request: in/ holds a file with a request's name and length that cp_prove may not read (Permission denied). It stays there without a response; once the account that runs cp_prove may read it, the next scan takes it like any other file there. Printed once in a process: look into in/ (ls -l) for later ones; the service goes on (prover/docs/ends-in-short.md)\n"))
    stop(End::WriteFailed);
}

// Prints a batch's log lines (`books`) after its release. `bears`: the time set on its files, ms.
inline void Prover::close_the_books(const Proved &b, long long bears) {
  const unsigned long long number = batch.number++;
  if (!number && closed_front.on && !closed_front.print_lines()) stop(End::WriteFailed);
  const BatchTimes &t = b.times;
  // A production build prints only the batch's number, its leaf table's number, which is in every
  // proof of the batch, and its number of requests: how long the opening waited for the keeper
  // unit's loader thread follows the tokens' routing. A test build adds the release's time and the
  // loader's three times.
  if (!SERVING_OR(closed_front.say_closed("batch %llu leaf %u n %d\n", number, t.leaf, b.B),
                  closed_front.say_closed("batch %llu leaf %u n %d tick %lld.%03lld loader_read %.3f loader_rebuild %.3f "
                                          "loader_waited %.3f\n",
                                          number, t.leaf, b.B, bears / 1000, bears % 1000,
                                          t.loader_read, t.loader_rebuild, t.loader_waited)))
    stop(End::WriteFailed);
  if (args.measure &&
      !closed_front.say_closed("measure: batch %llu members %d in_step %d leaf %u start %.3f head %.3f waves %.3f fold_begin %.3f "
                        "fold_walk %.3f fold_finish %.3f claims %.3f staged %.3f tick %lld.%03lld coins %s weights "
                        "%s bytes %zu join %.3f stage %.3f end %.3f file_write %.3f file_sync %.3f file_make %.3f file_lock %.3f fold_next %.3f loader_read %.3f loader_rebuild %.3f loader_waited %.3f cause %d\n",
                        number, b.B, args.in_step, t.leaf, t.start, t.head, t.waves, t.fold_begin,
                        t.fold_walk, t.fold_finish, t.claims, t.staged, bears / 1000, bears % 1000,
                        args.test_coins.empty() ? "drawn" : "test", args.weights.c_str(),
                        t.member_bytes, t.join, t.stage, t.end, t.file_write, t.file_sync, t.file_make,
                        t.file_lock, t.fold_next, t.loader_read, t.loader_rebuild, t.loader_waited, (int)t.cause))
    stop(End::WriteFailed);
}

// Implements `Work`, the interface of serve_the_spool() (service_loop.h describes each function).
struct ServiceWork {
  Prover &P;
  Prover::Proved last; // the batch that prove() proved last
  bool books_of_the_rider = false; // while `prove` releases a rider: the books are that rider's
  void the_loop_offers(const ServiceSteps *steps) {
    P.take_ahead = steps->take_ahead;
    P.release_now = [this, steps] {
      books_of_the_rider = true;
      steps->release();
      books_of_the_rider = false;
    };
  }
  bool opening_rides() const { return !P.riders.empty(); }
  static constexpr int CUT_SHORT = (int)End::CutShort;
  void a_request_is_unreadable() { P.say_the_alarm(End::Unreadable); }
  void look_for_the_word_to_end() { stop_if_asked(); }
  bool must_end() { return P.isk_fault != nullptr || P.self_check_failed != nullptr || P.a_live_leaf_is_spent(); }
  void seat(std::vector<Spool::Request> &taken) { P.seat_the_batch(taken); }
  // Prints where the time before the proving went, in seconds. "names": the last release's naming.
  // "sync" and "removal" are the spool's syncing thread's latest round (Spool::settle_for_ever()).
  void the_head_took(long long take, long long release, long long books, long long seat) {
    if (P.args.measure &&
        !closed_front.say_closed("measure: head take %.3f release %.3f names %.3f sync %.3f removal %.3f books %.3f seat %.3f\n", take / 1e3,
                                 release / 1e3, P.spool.naming_ms / 1e3, P.spool.syncing_ms / 1e3, P.spool.removing_ms / 1e3,
                                 books / 1e3, seat / 1e3))
      stop(End::WriteFailed);
  }
  int prove() {
    last = Prover::Proved{true, P.batch.B};
    last.times.start = epoch();
    return P.prove_the_batch(last.times);
  }
  void note(int alarm) {
    P.say_the_alarm((End)alarm);
    if (!P.riders.empty()) // that batch is the youngest rider: its cause is settled when it ends (end_the_batch())
      P.riders.back()->ending.alarm = (End)alarm;
    else if (last.times.cause == End::None)
      last.times.cause = (End)alarm;
  }
  void close_the_books(long long bears) { P.close_the_books(books_of_the_rider ? P.rider_proved : last, bears); }
  void alarm_without_a_batch(int alarm) {
    P.say_the_alarm((End)alarm);
    SERVING_OR(, printf("# alarm %d (prover/ends.h) without a batch\n", alarm));
  }
  [[noreturn]] void the_spool_failed() {
    printf("the spool: %s\n", P.spool.why);
    stop(End::WriteFailed);
  }
};

// Opens the spool directory (-spool) or one batch's directories (-in, -out) early: a wrong one, or
// one that another process serves, is rejected before the long set-up. Returns 0 or 2.
inline int Prover::open_the_spool() {
  // The vocabulary size comes from the parameter file (`policy`): plan_memory() sets Vv only later.
  if (!(args.one_batch() ? spool.open_one_batch(args.in, args.out, NT, (uint32_t)policy["V"]) : spool.open(args.spool, NT, (uint32_t)policy["V"]))) {
    printf("refused (cp-spool): %s\n", spool.why);
    return refused("cp-spool");
  }
  if (spool.took_over) printf("the folder named with -out held only .tmp of a batch that ended early: what was staged there and never released has been removed\n");
  // Two additions to stop() (ends.h). First (g_end_first), before any key or mask is zeroed,
  // Spool::shut() sets a flag, without allocating: a thread that runs on during the wipe can begin
  // no naming in out/. As the wipe's last step (g_end_wipe), what is staged is removed, so that a
  // response has its final name or does not exist; the rest is left for cp_prove -clear or for the
  // next process on the same directory to erase. The secrets are wiped first, as the removal
  // allocates and memory may be what ran out.
  static Spool *const the_spool = &spool;
  g_end_first = [] { the_spool->shut(); };
  static bool (*const wipe_without_the_spool)() = g_end_wipe;
  g_end_wipe = [] {
    bool zeroed = !wipe_without_the_spool || wipe_without_the_spool();
    try {
      zeroed = the_spool->drop_what_is_staged() && zeroed; // false: something may be left unchanged under out/.tmp/; the exit then counts as "not all wiped"
    } catch (...) {
      zeroed = false;
    }
    return zeroed;
  };
  return 0;
}

// Serves until the stop file, -most-batches, -end-when-empty, the end of a test's randomness file
// or a stop caused by isk (such as a full leaf table); with -in, one batch. Returns 0 or a status.
inline int Prover::serve() {
  if (args.measure && !closed_front.say_closed("measure: ready %.3f process_start %.3f batch %d in_step %d\n", epoch(), process_start,
                                               args.batch, args.in_step))
    stop(End::WriteFailed);
  ServiceWork work{*this};
  ServiceTerms terms{args.batch, args.most_batches, args.end_when_empty, args.overlap};
  if (args.one_batch()) terms.most_batches = 1, terms.end_when_empty = true;
  // A test's randomness file (-test-coins) covers only this many batches.
  if (test_coins_batches >= 0 && (!terms.most_batches || terms.most_batches > test_coins_batches))
    terms.most_batches = test_coins_batches;
  if (int rc = serve_the_spool(work, spool, terms)) return rc;
  spool.tidy();
  // A stop caused by isk; the last line names it. If a batch met it, its requests have "failed".
  if (self_check_failed) isk_fault = self_check_failed; // not reached as the calls are ordered: end_the_batch() ends the process first. It guards against a change of that order
  if (isk_fault) {
    name_the_end(isk_fault); // for the last line
    return 2;
  }
  forget_the_opening_library();
  fork_memory.report("at the end");
  (void)fork_memory.ledger("at the end");
  say_of_the_memory();
  return 0;
}

// Calls setup_opening() and records how much GPU memory it takes (fork_memory.taken_by_others).
inline int Prover::set_up_the_opening_library() {
  const size_t before = ForkMemory::gpu_bytes_in_use();
  if (int rc = setup_opening()) return rc;
  fork_memory.taken_by_others["the opening library's set-up"] = std::max(ForkMemory::gpu_bytes_in_use(), before) - before;
  return 0;
}

// The whole program: set-up, then the service. end_of_main() turns the result into the exit code.
inline int Prover::run(int argc, char **argv) {
  this->argc = argc;
  this->argv = argv;
  if (int rc = read_inputs()) return rc;
  // A production build prints no time here: open_the_spool() takes longer the more requests were
  // left in work/. Every build prints setup_opening()'s duration, which depends on no request.
  auto stamp = [&](const char *what) {
    SERVING_OR(printf("start-up: %s\n", what), printf("start-up: %.3f s since the epoch: %s\n", epoch(), what));
    fflush(stdout);
  };
  stamp("arguments and parameter file read");
  if (int rc = the_ranges_are_guarded()) return rc;
  if (int rc = open_the_spool()) return rc;
  if (int rc = set_up_the_opening_library()) return rc;
  stamp("the weight table's keepers set up");
  if (int rc = test_the_coins()) return rc;
  // THE FRONT CLOSES HERE, AND NOWHERE ELSE: the next line reads the first secret. After close()
  // a production build prints only fixed lines. Read so far: no weight; of isk only isk.txt and the
  // configuration files; of requests those left in work/, zeroed at once.
  // A check of the sources that is not part of this repository tests this, only for the ways of
  // reading a secret that it lists. Printed so far of clients' files: see ClosedFront::close().
  // With -public-weights read_isk() has already derived the seed from that option's file: under
  // that option the seed is public.
  if (!closed_front.close()) {
    printf("cp_prove: the output cannot be written, or /dev/null cannot be opened\n");
    return refused("write-failed");
  }
  if (int rc = read_the_seed()) return rc;
  stop_if_asked();
  if (int rc = make_tables()) return rc;
  stop_if_asked();
  if (int rc = plan_memory()) return rc;
  stamp(args.stream_weights ? "weights read and pinned" : "weights ready");
  stop_if_asked();
  if (int rc = setup_g_forks()) return rc;
  stop_if_asked();
  if (args.close_on_the_slab > 0) // the attention objects and the head leave the slab's first bytes free
    fork_memory.single_base =
        std::min(ForkMemory::rounded((size_t)(args.close_on_the_slab * 1e9)), fork_memory.arena_first);
  if (int rc = setup_attention_forks()) return rc;
  stop_if_asked();
  if (int rc = setup_head_fork()) return rc;
  stop_if_asked();
  if (int rc = plan_stream_ids()) return rc;
  if (int rc = setup_buffers()) return rc;
  stop_if_asked();
  if (int rc = build_header()) return rc;
  stop_if_asked();
  if (int rc = plan_sections()) return rc;
  if (int rc = chain_make()) return rc;
  if (int rc = set_up_the_shared_chain()) return rc;
  if (int rc = read_the_counts()) return rc; // uses the shared chain's number of columns per batch
  hold_the_close_to_its_room(); // the GPU scratch memory (the slab) grows here if it must, not after the first batch pass
  if (int rc = set_up_the_seats()) return rc;
  fork_memory.plan_repeats();
  stop_if_asked();
  fork_memory.report("before the first batch");
  (void)fork_memory.ledger("before the first batch");
  stamp("set-up done");
  stop_if_asked();
  if (int rc = serve()) return rc;
  // The run has reached its last line. The rest is the same at every exit, in stop() (ends.h).
  g_end_at_the_last_line = true;
  if (!opening_library_forgot_all) g_zeroing_failed = true;
  // Only in a test's run (inputs.cuh, LENT_POISON): shows the test that the overwriting took place.
  SERVING_OR(, if (blk::card_test_byte >= 0) printf("a test's byte %02x (LENT_POISON) was laid over the GPU engine's buffers %llu times, %llu bytes in all\n", (unsigned)blk::card_test_byte, blk::card_fills.load(), blk::card_bytes_filled.load()));
  printf("the last line: %.3f s since the epoch: every response that the service owed has been released\n", epoch());
  return 0; // success; `end_of_main` (ends.h) makes the exit
}
