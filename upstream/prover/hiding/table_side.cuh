// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The first three sumcheck rounds of three product layers on the table side of LKP, the lookup
// argument by product trees (blocks/iop/lkp.h): the leaf layer, the bit layer and the last layer of
// the product tree, here called the bottom layer. Their tables L and R: see tside::Masks.
//
// The rounds bind the top three bits of a layer's index, whose 8 values are called slabs. Each
// round is one pass over the multiplicity bits, packed into mask bytes (bit masks): bit a for
// slab a. Folded tables are not stored until the caller of tside::mask_door() writes them after the
// third round: a round recomputes a folded entry as sum_h E[h] X_h from entries X_h of the unfolded
// table, with E[h] = eq(the challenges so far, h). The one-block kernel k_end combines the blocks'
// partial sums into g(0) and the leading coefficient of the round polynomial g. It passes both,
// unmasked, to sc_door::round() (prover/hiding/sumcheck_door.cuh), which sends the round's masked
// message and returns the challenge.
//
// The caller must begin the layer's sumcheck (k_layer_begin, prover/hiding/product_test.cuh)
// before tside::mask_door(); otherwise sc_door::round() traps. Only the bit layer has such a caller
// (ForestTable::bit_layer(), prover/hiding/forest_table.cuh). Nothing in this tree calls
// tside::mask_door() for the leaf layer or the bottom layer. k_leaf sums L R, without the 1 of
// LKP's leaf summand 1 + L R.
//
// Not self-contained: prover/hiding/product_test.cuh includes this file after St, Chal<F>, pt_hi(),
// g_sms and the other names used here. tside::PF forwards to the field's routines
// (prover/hiding/pt_field.h): each coordinate of a product in the cubic extension field is a sum of
// products that is reduced modulo the prime once (the first source below).
//
// Public sources:
//   Delayed modular reduction: J-G. Dumas, P. Giorgi, C. Pernet, "Dense Linear Algebra over
//   Word-Size Prime Fields: the FFLAS and FFPACK packages" (arXiv:cs/0601133), section 2.1.
//   cub::BlockReduce, cub::WarpReduce: NVIDIA, CUB documentation
//   (https://nvidia.github.io/cccl/cub/).
//   Blocks' results added by a second kernel: M. Harris, "Optimizing Parallel Reduction in CUDA"
//   (NVIDIA Developer Technology).
//   NVIDIA, CUDA C++ Programming Guide: "Synchronization Functions" (__syncwarp).
//   See also: NVIDIA, CUDA Programming Guide, "C/C++ Language Extensions",
//   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html:
//   "Warp Synchronization Function" (__syncwarp: at each call here one warp alone shares the data).
#pragma once
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include "hiding/table_side_common.inc"
#include "hiding/table_side_kernels.inc"
