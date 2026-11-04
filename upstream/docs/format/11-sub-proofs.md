*Part of [the file format](FORMAT.md). Rule numbers and terms are those of that page.*

## 11. The sub-proofs of a segment

A proof's first chain walks the model segment by segment (R14.4). Section 10 gives what the frame of
a segment sends and samples. This section gives what the **sub-proof** of each kind of segment
places between the frame's rounds: every round by name, how many scalars, roots and answers the
prover sends in it, and what is sampled. It is written from the programs' source. This section names
no source for the methods that the sub-proofs use (sumchecks, lookups, tests on the size of
committed values and others). `PUBLISHED-WORKS.md` does: its section "The paper that this protocol
builds on" says what the programs have of that paper, and its section "Published methods of the
proof system" lists the publications that comments of the programs' source cite, each with files
that name it.

Part 11.1 gives the kinds of step that all sub-proofs share, 11.2 a feed-forward segment, 11.3 an
attention segment and 11.4 the output head. Part 11.5 gives how the weights are placed in the stored
table. The segments of a request's first chain come in this order: G_0, A_0,
G_1, A_1, .., H. G_g is the feed-forward segment of the layers of group g, A_g the attention segment
of that group, and H the output head. Outside the segments a model's sub-proofs send nothing (R14.4
(e)).

Each of the parts 11.2 to 11.4 has two halves: the first half gives the rounds, and the second half
gives the content. Part 11.1 has three pieces: the first gives the rounds, and the second and the
third give the content. The content is what each committed cell and each sent scalar holds, how the
keys of the lookups are formed, the tests on the public numbers with the no-overflow conditions, and
every claim as a formula, in the order of the claims. This section does not give the arguments why
the claims imply the model's integer function, an honest prover's choices where a verifier accepts
several values (such as the limbs of a number), or how a model's tensors become the committed
integers (11.5; section 17).

The parts of this section are files of their own:

- 11.1 The machinery: [11-1-machinery.md](11-1-machinery.md)
- 11.2 A feed-forward segment: [11-2-feed-forward-segment.md](11-2-feed-forward-segment.md)
- 11.3 An attention segment: [11-3-attention-segment.md](11-3-attention-segment.md)
- 11.4 The output head: [11-4-output-head.md](11-4-output-head.md)
- 11.5 How the weights are placed in the stored table:
  [11-5-weights-in-the-table.md](11-5-weights-in-the-table.md)
