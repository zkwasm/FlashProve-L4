#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Reads the two folders of one batch, cp_prove's -in and -out, for audit.py --batch.

A batch's first key, from which its private randomness derives, depends on every slot: on the
request's words, or on "no request" (docs/format/COIN-KEYS.md, "2. The schedule"). Nothing
that is kept for the audit (kept.py) records the slots. read() derives them from the folder of
-in, which must be as it was when cp_prove read it.

How cp_prove assigns slots (prover/prover/spool.h: requests_in(), take_the_folder(),
read_request(); prover/prover/isk.cuh: read_isk()):
  - A request is a regular file <name>.req; <name> has 1 to 64 characters out of A-Z a-z 0-9 _ -.
  - Slot j is the j-th <name> in bytewise order, compared without ".req": "a" comes before "a-b".
  - cp_prove rejects a folder that holds anything else, no request, or more requests than a batch
    has slots (`most`); read() raises Unfit.
  - A file that is not a well-formed request (words_of()) is "no request" and keeps its slot.

Each folder is opened once, and each file is read at most once, through the folder's descriptor.
Nothing here prints.
"""
import hashlib
import os
import re
import stat
import struct

import kept

NAME = re.compile(r'[A-Za-z0-9_-]{1,64}')


class Unfit(RuntimeError):
    """Its message contains only file names and counts and may be printed."""


class Folder:
    def __init__(s, path, what):
        s.what = what
        try: s.fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        except OSError: raise Unfit('the folder of %s is missing or cannot be opened' % what) from None
        s.names = sorted(os.listdir(s.fd))

    def regular(s, n):
        try: return stat.S_ISREG(os.stat(n, dir_fd=s.fd, follow_symlinks=False).st_mode)
        except OSError: return False

    def read(s, n):
        try: f = os.open(n, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=s.fd)
        except OSError: raise Unfit('a file in the folder of %s cannot be opened: %s' % (s.what, kept.shown([n]))) from None
        with os.fdopen(f, 'rb') as h:
            if not stat.S_ISREG(os.fstat(f).st_mode): raise Unfit('what was opened in the folder of %s is no regular file: %s' % (s.what, kept.shown([n])))
            return h.read()


def words_of(b, T, V):
    """Returns the request's words, or None for "no request".

    A request is T + 2 little-endian 32-bit words: T token ids below V, the vocabulary size, then
    r and e with r < e <= T.
    """
    if len(b) != 4 * (T + 2): return None
    w = list(struct.unpack('<%dI' % (T + 2), b))
    return w if all(t < V for t in w[:T]) and w[T] < w[T + 1] <= T else None


def read(path_in, path_out, T, V, most):
    """Returns (slots, files, digest).

    slots: (name, words_of()'s result, True if the folder of -out has <name>.y and <name>.proof)
        per slot, in order.
    files: 'in/<name>.req', 'out/<name>.y', 'out/<name>.proof' -> bytes.
    digest: SHA-256 over the slots in order: the name's length, the name, the request file's
        length, the file. Each length is 8 bytes, little-endian.
    Raises Unfit if a file cannot be opened or a folder is not what one call of cp_prove reads and
    leaves.
    """
    fin, fout = Folder(path_in, '-in'), Folder(path_out, '-out')
    odd = [n for n in fin.names if not (n.endswith('.req') and NAME.fullmatch(n[:-4]) and fin.regular(n))]
    if odd: raise Unfit('the folder of -in holds %d name(s) that are no regular file <name>.req: %s' % (len(odd), kept.shown(odd)))
    names = sorted(n[:-4] for n in fin.names)
    if not names: raise Unfit('the folder of -in holds no request')
    if len(names) > most: raise Unfit('the folder of -in holds %d requests; a batch has at most %d slots' % (len(names), most))
    ends = {n: {e for e in ('.y', '.proof', '.failed') if n + e in fout.names} for n in names}
    odd = [n for n in fout.names if n not in {m + e for m in names for e in ends[m]} or not fout.regular(n)]
    if odd: raise Unfit('the folder of -out holds %d name(s) that are no regular file <name>.y, <name>.proof or <name>.failed of a request in the folder of -in: %s' % (len(odd), kept.shown(odd)))
    odd = [n for n in names if ends[n] not in ({'.y', '.proof'}, {'.failed'})]
    if odd: raise Unfit('%d request(s) have neither a reply and a proof nor «.failed» alone in the folder of -out: %s' % (len(odd), kept.shown(odd)))
    files, slots, h = {}, [], hashlib.sha256()
    for n in names:
        b = files['in/%s.req' % n] = fin.read(n + '.req')
        h.update(struct.pack('<Q', len(n)) + n.encode('ascii') + struct.pack('<Q', len(b)) + b)
        slots.append((n, words_of(b, T, V), '.proof' in ends[n]))
        if slots[-1][2]:
            for e in ('.y', '.proof'): files['out/' + n + e] = fout.read(n + e)
    odd = [n for n, w, proved in slots if proved and w is None]
    if odd: raise Unfit('%d file(s) in the folder of -in are no request, and the folder of -out has a proof for them: %s' % (len(odd), kept.shown(odd)))
    return slots, files, h.hexdigest()
