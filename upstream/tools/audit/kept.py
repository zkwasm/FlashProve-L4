#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""The files that keep a run's fresh random bytes for one audit (tools/audit/audit.py).

With `-keep-for-the-audit`, `cp_commit` writes `audit/key.fresh` and `audit/entry.<k>.fresh` into
the secret key directory `<isk>` (`folder` in messages), and `cp_prove` writes `audit/batch.fresh`.
Each contains the 32 bytes from the operating system that went into the private randomness of the
key file (`ivk/key.bin`), of leaf file k (`ivk/leaf/<k>.bin`; `entry` in names and messages) or of
one batch's proofs.
Kept.destroy() and KeptBatch.erase() unlink only these names, in `<isk>/audit`. They remove no
directory and use no pattern. They destroy a file's present blocks and its name: journals,
copy-on-write, snapshots, flash remapping and backups may still contain the bytes.
What has been tested: tools/audit/README.md, "What has been run"."""
import os
import re
import stat

NAME = re.compile(r'(key|entry\.(0|[1-9][0-9]*))\.fresh')


class Stop(RuntimeError):
    """Its message contains no file's contents and may be printed."""


def _is_plain(st): return stat.S_ISREG(st.st_mode) and st.st_size == 32 and (st.st_mode & 0o777) == 0o600 and st.st_uid == os.geteuid() and st.st_nlink == 1  # one name alone: a second hard link is another name for the same bytes


# A commitment's kept files (32 zeros each under a test seed). Anyone who has their bytes and the
# seed (`isk/seed.bin`) can compute every mask in the key file and the leaf files. So they may be
# kept, for one audit, only if: no later run that creates a key file or a leaf file reads
# them; the audit sends nothing and draws no random value; its report lists places, not values;
# nothing of the commitment is released before the audit's YES; and the kept files are durably
# destroyed before that YES.
# `<isk>/audit` is opened once, and the files are reached only through the descriptor that
# `_handle()` returns (dir_fd=None would mean the current working directory).
# `copies`: the leaf files' numbers k. `odd`: why `audit/` is unfit, or None.
class Kept:
    def __init__(s, isk, copies):
        s.path = os.path.join(isk, 'audit'); s.who = ['key'] + list(copies); s.wanted = ['key.fresh'] + ['entry.%d.fresh' % k for k in copies]; s.bytes = {}; s.ino = {}; s.fd = None; s.odd = None
        try: s.fd = os.open(s.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError: pass
        except OSError: s.odd = 'isk/audit cannot be opened as a folder of its own (a link, a file, or closed to the caller)'
        if s.fd is not None:
            st = os.fstat(s.fd)
            if (st.st_mode & 0o777) != 0o700 or st.st_uid != os.geteuid(): s.odd = 'isk/audit/ is not a folder of mode 0700 that belongs to the caller'  # the nine permission bits: under a set-group-ID parent a directory inherits that bit

    def _handle(s):
        if s.fd is None: raise Stop('there is no folder isk/audit/')  # never a call with dir_fd=None
        return s.fd

    def _plain(s, name):
        try: return _is_plain(os.stat(name, dir_fd=s._handle(), follow_symlinks=False))
        except FileNotFoundError: return False

    # Reads no file. 'whole': `audit/` has mode 0700 and contains exactly the names in `wanted`,
    # each a regular 32-byte file of mode 0600 with one hard link, and the caller owns them all.
    def state(s):
        if s.odd: return 'not whole', s.odd
        if s.fd is None: return 'nothing', 'no folder isk/audit/'
        names = sorted(os.listdir(s.fd))
        if not names: return 'nothing', 'the folder isk/audit/ is empty'
        odd = [n for n in names if not NAME.fullmatch(n)]
        if odd: return 'not whole', '%d file(s) in isk/audit/ are named neither key.fresh nor entry.<k>.fresh%s' % (len(odd), ' (batch.fresh is a batch\'s kept file: audit_batch.sh)' if 'batch.fresh' in odd else '')
        lack = [n for n in s.wanted if n not in names]
        if lack: return 'not whole', 'missing: ' + ', '.join(lack)
        more = [n for n in names if n not in s.wanted]
        if more: return 'not whole', 'kept for an entry that does not stand in ivk/leaf/: ' + ', '.join(more)
        bad = [n for n in names if not s._plain(n)]
        if bad: return 'not whole', 'not a plain 32-byte file of mode 0600 with one name that belongs to the caller: ' + ', '.join(bad)
        return 'whole', '%d files' % len(names)

    def read(s):
        for n, who in zip(s.wanted, s.who):
            f = os.open(n, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=s._handle())
            try:
                st = os.fstat(f)
                if not _is_plain(st): raise Stop('isk/audit/%s: what was opened is not the plain file that was seen' % n)
                s.ino[n] = (st.st_dev, st.st_ino); b = bytearray(32)
                if os.readv(f, [b]) != 32: raise Stop('isk/audit/%s: a short read' % n)
            finally: os.close(f)
            s.bytes[who] = b
        return s.bytes

    # The bytes objects that level1.py and level1_proof.py make from these buffers cannot be wiped.
    def wipe(s):
        for b in s.bytes.values():
            for i in range(len(b)): b[i] = 0

    def destroy(s, count):
        """Overwrites with zeros, syncs and unlinks each file, then syncs the directory.

        `audit/` and the files must not be symbolic links, and no file may have a second hard link;
        links above `audit/` are followed (audit.py prints the real path).
        Raises Stop if the directory or a file is not as state() saw it (checked before the first
        file is touched), if an opened file is not the one that read() read, on a short write, if a name
        no longer leads to the file that was overwritten, or if
        `audit/` is not empty at the end. Adds 1 to count['destroyed'] for each file unlinked.
        A failure part way leaves some files unlinked and perhaps one more overwritten; audit.py
        then prints NOT WHOLE, never YES."""
        fd = s._handle()
        if s.odd: raise Stop(s.odd)
        here, there = os.fstat(fd), os.stat(s.path, follow_symlinks=False)
        if (here.st_dev, here.st_ino) != (there.st_dev, there.st_ino): raise Stop('isk/audit/ is no longer the folder that was opened')
        if sorted(os.listdir(fd)) != sorted(s.wanted): raise Stop('isk/audit/ holds other names than it did')
        for n in s.wanted:
            if not s._plain(n): raise Stop('isk/audit/%s is no longer the plain file that was seen' % n)
        for n in s.wanted:
            f = os.open(n, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                st = os.fstat(f)  # of the open file, not of its name
                if not _is_plain(st) or s.ino.get(n) != (st.st_dev, st.st_ino): raise Stop('isk/audit/%s: what was opened is not the file that was read' % n)
                if os.write(f, bytes(32)) != 32: raise Stop('isk/audit/%s: a short write' % n)
                os.fsync(f)
                at = os.stat(n, dir_fd=fd, follow_symlinks=False)  # a second look, as in KeptBatch.erase(): the name must still lead to the file that was overwritten
                if (at.st_dev, at.st_ino) != (st.st_dev, st.st_ino): raise Stop('isk/audit/%s: the name no longer leads to the file that was overwritten' % n)
            finally: os.close(f)
            os.unlink(n, dir_fd=fd); count['destroyed'] += 1
        os.fsync(fd)
        if os.listdir(fd): raise Stop('isk/audit/ is not empty behind the destruction')


SHOWN = re.compile(r'[A-Za-z0-9._-]{1,64}')
def shown(names):
    """File names joined for a printed line; a name that SHOWN does not match is only counted."""
    fit = [n for n in names if SHOWN.fullmatch(n)]
    return ', '.join(fit + (['%d more whose names are not printed' % (len(names) - len(fit))] if len(fit) < len(names) else []))


class KeptBatch(Kept):
    """A batch's kept file, `<isk>/audit/batch.fresh`.

    Anyone who has its 32 bytes and the seed can compute every mask, pad and salt of that batch's
    proofs. state() returns 'nothing' also if `audit/` contains files and none is named batch.fresh:
    they are not a batch's (a commitment's kept files, for example), and audit.py then erases
    nothing. Otherwise audit.py calls erase() at every verdict (YES, NO, NOT WHOLE), unless run
    with --rehearsal."""

    def __init__(s, isk):
        super().__init__(isk, []); s.who, s.wanted, s.seen = ['batch'], ['batch.fresh'], None

    def state(s):
        if s.fd is None: return ('not whole', s.odd) if s.odd else ('nothing', 'no folder isk/audit/')
        names = sorted(os.listdir(s.fd))
        if not names: return 'nothing', 'the folder isk/audit/ is empty'
        if 'batch.fresh' not in names: return 'nothing', 'isk/audit/ holds %d name(s), and none is batch.fresh' % len(names)
        try: st = os.stat('batch.fresh', dir_fd=s.fd, follow_symlinks=False); s.seen = (st.st_dev, st.st_ino, st.st_ctime_ns)  # the file seen here: `erase` touches no other. With the change time, because a removed file's inode number can be reused
        except FileNotFoundError: return 'not whole', 'isk/audit/batch.fresh was there and is gone'
        if s.odd: return 'not whole', s.odd
        if names != s.wanted: return 'not whole', 'isk/audit/ holds %d name(s) beside batch.fresh' % (len(names) - 1)
        if not s._plain('batch.fresh'): return 'not whole', 'isk/audit/batch.fresh is not a plain 32-byte file of mode 0600 with one name that belongs to the caller'
        return 'whole', '1 file'

    def erase(s, count):
        """Overwrites with zeros, syncs and unlinks one name, batch.fresh, then syncs the directory.

        Adds 1 to count['destroyed'] when the file is unlinked. Returns the names left in `audit/`:
        while there is one, `cp_prove` rejects the secret key directory (`cp-kept-for-the-audit`).
        Other names are left alone: `cp_prove` creates none there.
        batch.fresh is left alone too unless it is a regular file of at most 32 bytes that the
        caller owns and that has one hard link. A killed `cp_prove` can leave a shorter file; a
        longer one is not what `cp_prove` writes, and overwriting a very long one could fill the
        disk. Overwriting a file with a second hard link would also change the bytes under its
        other name, which may be outside `audit/`.
        It erases the file that state() saw and that read(), if called, read, or nothing. (After
        'not whole', as for a short file or another mode, audit.py calls erase() without read().)
        An audit can take minutes; meanwhile another process of the same account can move
        batch.fresh away or give its name to another file. Then the 32 bytes may still exist
        elsewhere, and a file that the audit has not looked at is not the audit's to overwrite:
        audit.py prints that the erasure stopped, never YES.
        Raises Stop in that case, if the file has changed since state() saw it, or if `audit/` was
        not opened, belongs to another account or is no longer the directory that was opened.
        Nothing has been touched in these cases but one, which the message states: the name changed
        hands between the overwrite and the unlink. A short write (Stop) or an OSError in the
        overwrite or the sync can also leave the file partly overwritten and not unlinked.
        If `audit/` was opened and the caller owns it, batch.fresh is erased even if the directory's
        mode is not 0700 (state() then returns 'not whole'): to leave the file would leave hiding
        void, and what is touched is bounded by the file's own owner, type, length, link count and
        identity, not by the directory's mode."""
        if s.fd is None and s.odd: raise Stop(s.odd)
        fd = s._handle()
        here, there = os.fstat(fd), os.stat(s.path, follow_symlinks=False)
        if (here.st_dev, here.st_ino) != (there.st_dev, there.st_ino): raise Stop('isk/audit/ is no longer the folder that was opened')
        if here.st_uid != os.geteuid(): raise Stop('isk/audit/ belongs to another account')
        n = 'batch.fresh'
        away, other = 'batch.fresh is no longer in isk/audit/: it was moved or removed while the audit ran, and this audit has not overwritten it', 'batch.fresh is no longer the file that the audit has looked at, or it was changed since (its inode\'s number or its time of change differs): it is left alone'
        if s.seen is None: raise Stop('batch.fresh went away while the audit first looked at isk/audit/; whatever has that name now is left alone')
        if s.ino.get(n, s.seen[:2]) != s.seen[:2]: raise Stop(other)
        mine = lambda st: stat.S_ISREG(st.st_mode) and st.st_size <= 32 and st.st_uid == os.geteuid() and st.st_nlink == 1
        same = lambda st, k: (st.st_dev, st.st_ino, st.st_ctime_ns)[:k] == s.seen[:k]
        try:
            st = os.stat(n, dir_fd=fd, follow_symlinks=False)
            if not same(st, 3): raise Stop(other)
            f = os.open(n, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd) if mine(st) else None
        except FileNotFoundError: raise Stop(away) from None
        except OSError: f = None  # it is there and cannot be opened for writing: left alone, and returned by name below
        if f is not None:
            try:
                st = os.fstat(f)  # of the open file, not of its name
                if not same(st, 3): raise Stop(other)
                good = mine(st)
                if good and os.write(f, bytes(st.st_size)) != st.st_size: raise Stop('batch.fresh: a short write')
                if good: os.fsync(f)
            finally: os.close(f)
            if good:
                try: still = same(os.stat(n, dir_fd=fd, follow_symlinks=False), 2)  # without the change time, which the overwrite has changed
                except FileNotFoundError: still = False
                if not still: raise Stop('the file that the audit has read is overwritten, but its name batch.fresh has changed hands since, and nothing was unlinked')
                os.unlink(n, dir_fd=fd); count['destroyed'] += 1
        os.fsync(fd)
        return sorted(os.listdir(fd))


def harden():
    """Tries to disable core files and dumps of this process and to lock its memory.

    Returns what succeeded and, in capitals, what failed. audit.py exits unless the first two
    succeeded."""
    got = []
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); got.append('no core file')
    except Exception: got.append('CORE FILES NOT SHUT OUT')
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        got.append('not dumpable' if libc.prctl(4, 0, 0, 0, 0) == 0 else 'STILL DUMPABLE')  # PR_SET_DUMPABLE
        got.append('memory locked' if libc.mlockall(3) == 0 else 'MEMORY NOT LOCKED (the system refuses: swap is not shut out)')  # MCL_CURRENT | MCL_FUTURE
    except Exception: got.append('STILL DUMPABLE, MEMORY NOT LOCKED (no prctl, mlockall)')
    return ', '.join(got)
