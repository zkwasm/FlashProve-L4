// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// T is an element of the extension field; F is that field's arithmetic. `Tr` is the sub-proofs'
// Fiat-Shamir transcript: a hash chain (blk::Chain) that the host lends to kernels
// (transcript_shell.cuh). Their sumchecks must send only through sumcheck_door.cuh.
#pragma once
#include "common/element_from_hash.h"
#include "hiding/pt_transcript.h"
#include <string>
#include <cmath>
#include "common/hash.h"
#include "common/side_streams.h"
typedef FE F;
typedef Ext T;
#include "hiding/psc_device.cuh"
// Which round of which block of the protocol a verifier's message belongs to: PSC (the sumcheck
// with a hiding polynomial), LINE (two claims into one), ZERO (a zero-check's point), BAT (batches
// linear claims), LKP (lookup), JOIN (claims of one shape into one, at a fresh point). round_name()
// (transcript_shell.cuh) gives each round's name in the proof's plan of rounds; LKP_PLANES and
// FOREST_POINT have JOIN's name.
enum class Round : unsigned char { PSC_EPSILON, PSC_ROUND, PSC_Y, LINE, ZERO, BAT_SIGMA, LKP, LKP_SHIFT, LKP_PLANES, FOREST_POINT, JOIN, PSC_VALUES };
#include "hiding/transcript_shell.cuh"
// Declares `name`, a reference to a Tr in shared memory, which the caller must fill by a copy
// (sc_door::tr_copy_by_lanes()). A byte array backs it because a __shared__ variable cannot have an
// initialiser, and blk::Chain's members have default ones.
#define TR_IN_SHARED(name)                                                                                             \
  __shared__ alignas(8) unsigned char name##_bytes[sizeof(Tr)];                                                        \
  Tr &name = *reinterpret_cast<Tr *>(name##_bytes)
#include "hiding/sumcheck_door.cuh" // the slots' mask entries (PADOFF, PAD_STRIDE) and every send of a hidden value
