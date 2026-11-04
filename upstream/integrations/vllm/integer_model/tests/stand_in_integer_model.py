# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for the package integer_model in the tests with vLLM: no weights, no model.

It offers the calls and attributes that the vLLM model class and its loader use, and raises
integer_model's own WindowFull and RangeConditionFailed.

Its "greedy choice" after a token is a hash of the sequence's tokens up to and including that token,
reduced modulo V - 2 (V: the vocabulary size). A token left out, fed twice or out of order changes
the following choices (each stays the same only with probability 1/(V - 2)). So one comparison tests
how token ids reach the package and choices come back: what vLLM serves must equal continuation(),
which computes the same choices without vLLM.

It raises Misuse when a sequence is started while another runs, when extend or free gets a sequence
other than the running one, and when extend gets no token id, or one that is not a Python int from
0 to V - 1.
"""

import hashlib
import json
import os
import time
import types

import torch

import integer_model
from integer_model import RangeConditionFailed, WindowFull


class Misuse(Exception):
    """The caller broke a rule of the interface."""


def choices(token_ids, vocabulary_size):
    """The stand-in's choice after each token of token_ids, a sequence's first tokens.

    The last two ids of the vocabulary are never chosen, so a test can give them special meanings
    (out_of_range_token and faulty_token).
    """
    digest, result = hashlib.sha256(), []
    for token in token_ids:
        digest.update(token.to_bytes(4, "little"))
        result.append(int.from_bytes(digest.digest()[:8], "little") % (vocabulary_size - 2))
    return result


def continuation(prompt, vocabulary_size, end_tokens, longest):
    """A greedy server's reply to the prompt. It stops after an end token or at the length limit."""
    reply = []
    while len(reply) < longest and not (reply and reply[-1] in end_tokens):
        reply.append(choices(prompt + reply, vocabulary_size)[-1])
    return reply


class Sequence:
    """The tokens that were fed so far."""

    def __init__(self):
        self.token_ids = []


class IntegerModel:
    """The stand-in for integer_model.IntegerModel.

    A fed token equal to out_of_range_token makes extend raise RangeConditionFailed. One equal to
    faulty_token makes it raise a RuntimeError, as a fault inside the package would.
    """

    def __init__(
        self,
        vocabulary_size,
        window_length,
        out_of_range_token=None,
        faulty_token=None,
        seconds_per_call=0,
    ):
        self.parameters = types.SimpleNamespace(V=vocabulary_size, T=window_length)
        self.out_of_range_token, self.faulty_token, self.seconds_per_call = (
            out_of_range_token,
            faulty_token,
            seconds_per_call,
        )
        self.default_dtype = torch.get_default_dtype()  # a record for the tests
        self.weights = None  # the loader reads this attribute
        self.running = None  # the one sequence that has not been freed
        self.calls = []  # a record for the tests

    def new_sequence(self):
        if self.running is not None:
            raise Misuse("new_sequence while another sequence is running")
        self.calls.append(("new",))
        self.running = Sequence()
        return self.running

    def extend(self, sequence, token_ids):
        if sequence is not self.running:
            raise Misuse("extend on a sequence that was freed or never created")
        token_ids = list(token_ids)
        if not token_ids or not all(
            type(t) is int and 0 <= t < self.parameters.V for t in token_ids
        ):
            raise Misuse(f"extend with {token_ids!r}, which is not a list of at least one token id")
        self.calls.append(("extend", token_ids))
        time.sleep(self.seconds_per_call)
        before = len(sequence.token_ids)
        if before + len(token_ids) > self.parameters.T:
            raise WindowFull()
        sequence.token_ids += token_ids
        if self.faulty_token in token_ids:
            raise RuntimeError("The stand-in was fed its faulty token.")
        if self.out_of_range_token in token_ids:
            raise RangeConditionFailed(before + token_ids.index(self.out_of_range_token), 1)
        return torch.tensor(
            choices(sequence.token_ids, self.parameters.V)[before:], dtype=torch.int64
        )

    def free(self, sequence):
        if sequence is not self.running:
            raise Misuse("free on a sequence that was freed or never created")
        self.calls.append(("free",))
        self.running = None


def load_from_folder(folder, device, options=None):
    """The vocabulary size comes from config.json in `folder`, and the rest from the options."""
    with open(os.path.join(folder, "config.json")) as f:
        vocabulary_size = json.load(f)["vocab_size"]
    return IntegerModel(vocabulary_size, **(options or {}))


_registered = False


def register():
    """Makes integer_model.load_from_folder load the stand-in when the options have the key
    "stand_in", and the package's own model otherwise. A second call does nothing.
    """
    global _registered
    if _registered:
        return
    _registered = True
    package_load_from_folder = integer_model.load_from_folder

    def stand_in_or_package(folder, device, options=None):
        if options and "stand_in" in options:
            rest = {name: value for name, value in options.items() if name != "stand_in"}
            return load_from_folder(folder, device, rest)
        return package_load_from_folder(folder, device, options)

    # The loader looks the function up at each load.
    integer_model.load_from_folder = stand_in_or_package
