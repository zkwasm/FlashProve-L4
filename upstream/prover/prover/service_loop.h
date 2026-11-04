// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The main loop of the prover program cp_prove: the order of a batch's steps, which
// prover/prover/service.cuh implements. Host code only, and a template over the prover (`Work`), so
// that it can run with a stand-in.
// Each iteration (step 4 only if step 1 took a request):
//   1. takes the next batch's requests from in/ of the spool directory (prover/prover/spool.h);
//   2. releases every staged response, that is, moves it from out/.tmp/ to its final name in out/:
//      those of the batch proved in the previous iteration, and the response "failed" for requests
//      that cannot be served or that a dead process left in work/;
//   3. prints that batch's log lines (`books` in identifiers);
//   4. gives each request a slot (its place in the batch), proves the batch, stages its responses.
// Step 1 comes first so that it does not wait for step 2, whose duration depends on the number of
// responses and on how many of them failed. Step 1 still follows the releases of earlier iterations
// and the staging inside prove(), so the moment at which it runs does depend on those durations.
// ServiceTerms::overlap (see ServiceSteps) changes when steps 1 to 3 run, not which steps run.
#pragma once
#include "prover/spool.h"
#include <ctime>
#include <functional>
#include <vector>

struct ServiceTerms {
  int seats = 1;               // the most requests in a batch
  long long most_batches = 0;  // end after this many batches have been taken; 0: no limit
  bool end_when_empty = false; // end when a batch is due and no request is waiting
  bool overlap = false;        // the prover is offered ServiceSteps
};

// With ServiceTerms::overlap, prove() can run steps 1 to 3 itself, so that a batch's opening (the
// part of its proofs about the committed weight table) can be finished while the next batch is
// proved (`cp_prove -overlap`). take_ahead() and release() may be called only from inside prove(),
// on its thread: they change the loop's variables without a lock. Only if take_ahead() returns at
// least 1 may prove() return before all of its batch's responses are staged, because the loop calls
// prove() again only for a batch that was taken. opening_rides() must then return true, and a later
// prove() must stage the rest and call release(). Several batches may wait in this way
// (`cp_prove -overlap-two`: two): release() is called once for each, oldest first. At each
// release() nothing of a younger batch may be staged except "failed" for requests that got no slot,
// because Spool::release() releases all that is staged. The loop checks none of this.
struct ServiceSteps {
  std::function<int()> take_ahead; // step 1 for the next batch; returns how many requests it took
  std::function<void()> release;   // step 2, and step 3 for the oldest batch that waits
};

// serve_the_spool() returns 0 after a stop condition: an entry named `stop` in the spool directory
// (removed afterwards only if it is an empty regular file: Spool::stop_is_obeyed()),
// ServiceTerms::most_batches or end_when_empty, or must_end().
// `Work` must provide:
//   void look_for_the_word_to_end();               called first in every iteration and again before
//                                                  prove(), to act on a pending stop such as a
//                                                  signal; may exit the process
//   bool must_end();                               true: no further batch is taken, ahead or not;
//                                                  what is proved is released, and the loop ends
//   void seat(std::vector<Spool::Request> &taken); gets the batch's requests, sorted by id
//                                                  (Spool::take()), and gives each its slot
//   int prove();                                   proves that batch and stages its responses; not
//                                                  0: a failure of the machine or the program;
//                                                  serve_the_spool() returns that value, releasing
//                                                  nothing
//   void note(int alarm);                          an alarm for the log lines of the batch that
//                                                  prove() proved last
//   void close_the_books(long long bears_ms);      step 3 for the batch just released;
//                                                  bears_ms: the time set on its responses' files
//   void alarm_without_a_batch(int alarm);         reports an alarm when no request was taken
//   [[noreturn]] void the_spool_failed();          ends the process; spool.why says what went wrong
//   void the_head_took(take, release, books, slot) milliseconds spent in steps 1, 2 and 3 and in
//                                                  giving the slots, for measurements
//   static constexpr int CUT_SHORT;                for note() and alarm_without_a_batch(): one scan
//                                                  did not read all names in in/ (Spool::cut_short)
//   void a_request_is_unreadable();                called right after each step 1 that left in in/
//                                                  a file that the service may not read
//                                                  (Spool::unreadable), whether or not requests
//                                                  were taken; it is no batch's alarm
//   void the_loop_offers(const ServiceSteps *);    called once, before the first iteration, only
//                                                  with ServiceTerms::overlap; the pointer is valid
//                                                  until serve_the_spool() returns
//   bool opening_rides();                          true while a batch waits for a later prove() to
//                                                  stage the rest: see ServiceSteps
template <class Work> int serve_the_spool(Work &work, Spool &spool, const ServiceTerms &terms) {
  auto steady_ms = [] { // a monotonic clock
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000 + t.tv_nsec / 1000000;
  };
  int batches_wait = 0;       // batches whose `prove` has returned, not yet released
  long long batches = 0;      // batches taken
  bool by_the_stop_file = false;
  // What step 1 found. `made`: the loop has not used it yet. `ended`: a stop condition was met.
  struct Taking {
    bool made = false, ended = false;
    bool more_lies_there = false; // in/ is too full for one scan: it does not count as empty
    bool left_unopened = false;   // slots stayed free, and files that may be requests were not opened: it does not count as empty either
    std::vector<Spool::Request> taken;
    long long began = 0, taken_at = 0;
  } next;
  auto take = [&] {
    next = Taking{};
    next.made = true;
    next.began = steady_ms();
    by_the_stop_file = spool.stop_is_asked();
    next.ended = by_the_stop_file || (terms.most_batches && batches >= terms.most_batches) || work.must_end();
    if (!next.ended && spool.take(next.taken, terms.seats) < 0) work.the_spool_failed();
    if (!next.ended && spool.unreadable) work.a_request_is_unreadable();
    next.more_lies_there = !next.ended && spool.cut_short;
    next.left_unopened = !next.ended && spool.left_unopened;
    next.taken_at = steady_ms();
  };
  // Steps 2 and 3.
  long long release_began = 0, released_at = 0, booked_at = 0;
  auto release = [&] {
    release_began = steady_ms();
    const long long bears = Spool::now_ms();
    spool.naming_ms = 0;
    if (spool.staged() && !spool.release(bears)) work.the_spool_failed();
    released_at = steady_ms();
    if (batches_wait) { // the oldest of them
      work.close_the_books(bears);
      batches_wait--;
    }
    booked_at = steady_ms();
  };
  const ServiceSteps steps{[&] {
                             take();
                             return (int)next.taken.size();
                           },
                           release};
  if (terms.overlap) work.the_loop_offers(&steps);
  for (;;) {
    work.look_for_the_word_to_end();
    if (!next.made) take();
    Taking now = std::move(next);
    next = Taking{};
    if (work.opening_rides()) // a later `prove` calls `release` (see ServiceSteps)
      release_began = released_at = booked_at = steady_ms();
    else
      release();

    if (now.taken.empty()) {
      if (now.more_lies_there) work.alarm_without_a_batch(Work::CUT_SHORT);
      if (now.ended || (terms.end_when_empty && !now.more_lies_there && !now.left_unopened)) break;
      usleep(50000);
      continue;
    }
    batches++;
    work.seat(now.taken);
    work.the_head_took(now.taken_at - now.began, released_at - release_began, booked_at - released_at,
                       steady_ms() - booked_at);
    spool.syncing_ms = 0, spool.removing_ms = 0; // the syncing thread's last round: reported once
    work.look_for_the_word_to_end();
    if (int status = work.prove()) return status;
    batches_wait++;
    if (now.more_lies_there) work.note(Work::CUT_SHORT);
    if (next.made && next.taken.empty()) next = Taking{}; // empty `take_ahead`: step 1 runs again
  }
  // Waits, without a time limit, until every released response is on disk.
  if (!spool.settled(-1)) work.the_spool_failed();
  if (by_the_stop_file) spool.stop_is_obeyed();
  return 0;
}
