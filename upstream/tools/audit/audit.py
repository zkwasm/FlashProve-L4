#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""Audits a commitment's public files, and proofs, against the provider's secrets (README.md).

Why: the verifier `cp_verify` also accepts files in which a bug made masks zero, left out a salt
or used a mask twice.
The audit prints counts, names, digests and verdicts, and no byte of a file, derived key or random
value; failed checks are lines `CRY: <count> x <what>`. Code and output say "folder" for a
directory, `world` for <files>, `entry` for a leaf file (`ivk/leaf/<k>.bin`), "stored table" for a
weight table and "stored copy" for a leaf table (the weight table of one leaf file).

usage: audit.py <tree> <pubtool's output directory> <files> <1|2> --level=0|1 [options]
  <tree>            the top of this repository
  <1|2>             the field; 2 is Goldilocks, the programs' only field
  <files>           a directory with config/{policy.txt,layout.txt,KL.bin}, ivk/key.bin,
                    ivk/leaf/<k>.bin, in/<n>.req, out/<n>.{proof,y}; for level 1 also seed.test,
                    coins.test, isk/leaf/<k>/fresh.bin, commit.contexts.txt, batch.contexts.txt
  --report=<new file>  also writes the tables, openings, notes and failed checks
  --proofs=a,b | -  these proofs only | none
  --ledger=<new file>  at level 1, writes each replayed draw's generator, stream, length, kind
                    and place, without values. Neither <new file> may exist yet. The directory of
                    each must exist, and no other account may be able to change that directory or a
                    place above it (others_can_change())
  --no-answers      self-test of level 1. The key file and each leaf file contain, after the weight
                    table's root, "answers", each a weighted sum over a row plus a public factor
                    times a random pad. Derives the commitment ID, and each leaf file's key and a
                    batch's process key (both are keys for random values), without the answers,
                    which the parser has already read: every value that depends on the ID or on
                    these keys must then fail its check. Rejected with --commitment and --batch
  --flip            self-test of level 1: flips one bit in each of a set of records, in the copy
                    that the comparisons read. Each flipped record that level 1 checks must fail,
                    and no other. The last `FLIP:` line gives the result, not the status (1 on
                    success)
  --drawn           the files were made without test options (reads isk/seed.bin,
                    isk/leaf/<k>/fresh.bin): only the salts of the weight tables' opened tree leaves
                    can be recomputed. Only then may level 1 get proofs of more than one batch
  --commitment      audits a commitment before it is handed out (key file and leaf files, level 1)
                    and, if every check passes, destroys the files kept for the audit (kept.py:
                    `isk/audit/key.fresh` and `entry.<k>.fresh`, which the committer
                    `cp_commit -keep-for-the-audit` wrote), but only with --erase-kept-files. With
                    --rehearsal: the same checks, and nothing is destroyed. One of the two must be given. Call it
                    through audit_commitment.sh
  --batch           audits one batch that the prover `cp_prove` made with -in, -out and
                    -keep-for-the-audit: its proofs at level 1; of the key file and leaf files only
                    what --drawn reaches. Then, whatever the verdict, erases the batch's kept file
                    (`isk/audit/batch.fresh`, 32 bytes), unless it is not what `cp_prove` writes or
                    has changed since the audit first looked at it; every case is listed in
                    kept.py, KeptBatch.erase(). If the file is not erased, the verdict is not YES.
                    All this only with --erase-kept-files; with --rehearsal it erases nothing; one of
                    the two must be given. Needs `-` for <files>, 2 for the field, --ivk=, --isk=,
                    --in= and --out=; each named directory must exist. Call it through audit_batch.sh
  --ivk= --isk= [--config=]  reads these directories where they are, with `-` for <files>. Never
                    copy a secret key directory (`isk`) for an audit: each copy would keep its own
                    count of a table's shown columns (`count.bin`), which has a limit. Without
                    --config: <isk>/config
status: 0 no check failed (--commitment, --batch: YES); 1 a check failed (NO); 2 the call is
        rejected (`refused`); 4 NOT WHOLE; 5 NOTHING KEPT HERE; 6 a rehearsal passed; 7 stopped
        without a verdict (also if its output cannot be written)
needs:  numpy; pubtool's output (pubtool.cpp: public parameters, from the verifier's code);
        libfield.so (g++ -O2 -fno-ipa-modref -shared -fPIC -o libfield.so fieldlib.cpp), beside
        field.py or named by AUDIT_LIB. audit_commitment.sh and audit_batch.sh make both. The
        library is loaded into the process that holds the secrets: name only your own build.

Every byte of the key file, leaf files and proofs is recorded under one class (the printed table
lists them), with a level:
  0  parsed: the hash chains (Fiat-Shamir transcripts) replay, and every opening hashes up to
     its root
  1  recomputed from the secrets and public data only, and compared by value
Exception: `base-word/main-message` (M* + lambda M, in a proximity test's last step) is not
recomputed at level 1: the recomputed mask M* is subtracted, and the rest, divided by lambda, is
compared with the last stage's committed row at its opened positions.
Whether a mask is applied, and not only committed, is not checked here: that rests on `cp_verify`
accepting the same bytes. Run `cp_verify` as well, and report both results.
Level 1 reaches part of the bytes; the printed table gives the share. What else a YES leaves
unchecked: README.md, "What a YES does not cover" and "What a batch's YES does not cover".
To destroy or erase a kept file is to overwrite it with zeros, sync it and unlink it. Journals,
copy-on-write, snapshots, flash remapping and backups may still contain its bytes, and not every
copy in this process's memory is wiped (kept.py)."""
import collections
import hashlib
import os
import re
import stat
import sys

sys.dont_write_bytecode = True
LEVEL2 = False  # level 2 is not in the tree (README.md)
ME = os.path.dirname(os.path.abspath(__file__))


# The look that run_suite.py makes before it works below a folder, each part to the letter:
# the constants and functions below are run_suite.py's own. Copy this block whole into a script that
# makes a folder, or works below one, that it was given; change nothing in it.
#
# To use it, call others_can_change(<the existing parent of the folder you will make>, or <the folder itself if it exists>).
# It returns None, or (place, reason): the first place at or above it, from / down, on the typed path, on the resolved
# path and through every link on the way, that another account may be able to change, that does not exist, or that could not be
# examined (a failed look is never a pass). On a find, print "<script>: " + refusal(<what>, <path>, place, why) on standard error and
# exit with status 2. run_suite.py passes its option --i-trust-every-account-on-this-machine as refusal()'s last argument,
# trust, and so adds that option to the same sentence; for a place that does not exist or could not be examined it prints another
# sentence. The advice to chmod is given only to a caller who is not root and owns that open folder. Then use the resolved
# path of the parent from there on. (run_suite.py also refuses its build folder, named by the environment variable BUILD,
# if that has a default access control list; that check is not part of this block.)
# It needs "import os", "import re" and "import stat" in the script. It brings its own printable(); a script that has
# one of its own keeps this one all the same.

ACL = "has a default access control list"
OPEN = "can be written by its group or by other accounts and is not sticky"
THEIRS = "belongs to another account"
UNSEEN = "could not be examined"
GONE = "does not exist"
INVISIBLE = "\u2800\u3164\u115f\u1160\uffa0\U00013441\U00013442\U0001d159"


def printable(text):
    """Returns `text` with each character that str.isprintable() rejects, and each one in INVISIBLE,
    written as its Python escape (such as \\x1b), so that a test's output cannot move the cursor or
    pass for this program's own."""
    return "".join(c if c.isprintable() and c not in INVISIBLE else ascii(c)[1:-1] for c in text)


def trusted_owners():
    """Returns the accounts whose directories this run may build and work in: this one, root, and
    under sudo (as root, with SUDO_UID set) the account that ran sudo, so that under sudo that
    account's own directories, such as its home, are not reported as another account's."""
    owners = {0, os.geteuid()}
    if os.geteuid() == 0 and re.fullmatch(r"[0-9]+", os.environ.get("SUDO_UID", "")):  # anything else is not trusted
        owners.add(int(os.environ["SUDO_UID"]))
    return owners


def others_can_change(folder):
    """Returns (place, why) for the first directory or symbolic link at or above `folder`, on its
    path as typed, as the system reads it, or through the symbolic links on it, that another account
    could change, that does not exist or that could not be examined; else None. A place that another
    account could change belongs to an account that trusted_owners() does not name, or is a
    directory that its group or others can write in without the sticky bit. There another account
    could swap the directory below it, or point the link elsewhere."""
    def prefixes(path):
        parts = [p for p in path.split("/") if p]
        return ["/"] + ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]

    read = {}  # each symbolic link whose target looked_in() read and followed: its identity -> that target

    def identity(about):  # with the target kept beside it: a link swapped in may reuse the inode and even the change time
        return about.st_dev, about.st_ino, about.st_ctime_ns

    def looked_in(path):
        """Returns the directories that the system passes through to follow `path`, each symbolic
        link on the way, and the place where `path` leads. A link's target is followed from the
        directory that contains the link, so a link to a link elsewhere brings that directory in
        too. After 40 links, where the system gives up, the rest counts as plain names. Each link
        whose target it read goes into `read`, by its identity and with that target, if the same link
        stood before and after the read."""
        whole = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        places, names, at, links = [], [p for p in whole.split("/") if p], "/", 0
        while names:
            name = names.pop(0)
            if name == ".":
                continue
            if name == "..":  # the parent as the system takes it, not as typed
                at = os.path.dirname(at)
                continue
            places.append(at)
            here = os.path.join(at, name)
            try:
                first = os.lstat(here) if links < 40 else None
                target = os.readlink(here) if first is not None and stat.S_ISLNK(first.st_mode) else None
                if target is not None:
                    again = os.lstat(here)  # a link swapped during the read is left unread: the loop below refuses it
                    target = target if identity(again) == identity(first) else None
            except OSError:  # not there, or not examined here: the loop below decides
                target = None
            if target is not None:
                places.append(here)  # the link's owner can point it elsewhere
                read[identity(first)] = target
                links += 1
                names = [p for p in target.split("/") if p] + names
                at = "/" if target.startswith("/") else at
            else:
                at = here
        return places + [at]

    for place in prefixes(os.path.abspath(folder)) + prefixes(os.path.realpath(folder)) + looked_in(folder):
        try:
            about = os.lstat(place)
        except FileNotFoundError:  # a missing one is never a pass, even if the folder exists (as typed: a symbolic link, then "..")
            return place, GONE
        except OSError:  # nor is one that could not be examined
            return place, UNSEEN
        if about.st_uid not in trusted_owners():
            return place, THEIRS
        if stat.S_ISLNK(about.st_mode):  # only a link that the look read, still leading where it led then
            try:
                same = read.get(identity(about)) == os.readlink(place)
            except OSError:
                same = False
            if not same:
                return place, UNSEEN
        if stat.S_ISDIR(about.st_mode) and about.st_mode & 0o022 and not about.st_mode & stat.S_ISVTX:
            return place, OPEN
    return None


def refusal(what, path, place, why, trust=""):
    """The one sentence for a folder that is refused because place above it may be changed by another account: what it
    is (such as "work folder"), its path, place and why as others_can_change() gave them, and trust, the words that a
    caller who may say so adds (the runner: its option; a script without one: nothing)."""
    return ("the %s %s lies below %s, which another account may be able to change (%s): it could put another folder in its "
            "place and choose what runs as you. %same a new %s below folders that only you or root can change (such as "
            "one in your home folder, or in /tmp, whose sticky bit keeps others from replacing it)%s; nothing was run"
            % (what, printable(path), printable(place), whose(place, why), cure(place, why), what, trust))


def whose(place, why):
    """Why another account may be able to change place, in words, for the refusals of the work folder and of BUILD."""
    if why == THEIRS:
        return "a symbolic link that belongs to another account" if os.path.islink(place) else "owned by another account"
    if why == ACL:
        return "a default access control list"
    if why == UNSEEN:
        return "it could not be examined"
    if why == GONE:
        return "it does not exist"
    return "its group or others can write in it, and it has no sticky bit"


def cure(place, why):
    """The first words of what to do about place: chmod go-w, for a folder of the caller's own that is open by its mode
    alone (its group may be the caller's private group, as under a umask of 002). Never for root, for whom a shared disk's
    mount point is «its own»: obeyed, it would shut every other account out."""
    try:
        mine = why == OPEN and os.geteuid() != 0 and os.lstat(place).st_uid == os.geteuid()
    except OSError:
        mine = False
    return ("If %s is yours, close it with chmod go-w (a further such folder below it, if any, is named in turn); "
            "otherwise n" % printable(place) if mine else "N")


def refuse(why):
    print('audit.py: refused: ' + why + (' Nothing was looked at or erased: if isk/audit/batch.fresh is there, it still is, hiding stays void for its batch, and none of its responses may be handed out.' if '--batch' in sys.argv else '')); sys.exit(2)


state = {'destroyed': 0, 'begun': False, 'commitment': False, 'batch': False, 'end': None}
# An error's message could contain a secret value.
def hook(kind, value, tb):  # prints the error's type and place, never its message; always exits
    try:
        while tb.tb_next: tb = tb.tb_next
        if state['end'] and issubclass(kind, Exception):  # --batch, once the kept file has been seen: an error ends as NOT WHOLE and erases that file; a keyboard interrupt continues below
            end, state['end'] = state['end'], None
            end('NOT WHOLE', 'the audit stopped (%s, %s:%d): an error of its own, or a file that is not what the format says' % (kind.__name__, os.path.basename(tb.tb_frame.f_code.co_filename), tb.tb_lineno))
        print('audit.py: NO VERDICT (status 7): the audit stopped (%s, %s:%d): an error of its own, an interrupt from the keyboard, or a file that is not what the format says.%s' % (kind.__name__, os.path.basename(tb.tb_frame.f_code.co_filename), tb.tb_lineno,
              (' None of this batch\'s responses may be handed out. ' + ('No kept file was touched: if isk/audit/batch.fresh was there, it still is, and until it is erased hiding is void for this batch. Fix the cause and run the audit again: with --erase-kept-files it erases the file at its end.' if not state['begun'] else
               'THE ERASURE HAD BEGUN (%d file is gone; if 0, batch.fresh may be overwritten and still be there). If batch.fresh is still in isk/audit/, run the audit again.' % state['destroyed'])) if state['batch'] else
              '' if not state['commitment'] else ' Nothing of this commitment may leave. No kept file was touched: mend the cause and run it again.' if not state['begun'] else
              ' THE DESTRUCTION HAD BEGUN (%d kept files are gone, one more may be overwritten): this commitment can no longer be audited: it counts as NO, and nothing of it may leave.' % state['destroyed']))
        sys.stdout.flush()
    finally: os._exit(7)
sys.excepthook = hook

if not __debug__: refuse('Python runs with -O or PYTHONOPTIMIZE, which switches the checks off. Run it without.')
words = [a for a in sys.argv[1:] if not a.startswith('--')]
opts = dict((a[2:].split('=', 1) + ['1'])[:2] for a in sys.argv[1:] if a.startswith('--'))
known = {'level', 'report', 'proofs', 'flip', 'no-answers', 'drawn', 'commitment', 'batch', 'rehearsal', 'erase-kept-files', 'ledger', 'config', 'ivk', 'isk', 'in', 'out'} | ({'pending', 'bend', 'omega', 'answer', 'trees'} if LEVEL2 else set())
levels = ('0', '1', '2') if LEVEL2 else ('0', '1')
USAGE = 'usage: audit.py <tree> <pubtool\'s output folder> <world> <1|2> --level=%s [options: the head of audit.py]' % '|'.join(levels)
valued = [a.split('=', 1)[0] for a in sys.argv[1:] if a.startswith('--') and '=' in a and a[2:].split('=', 1)[0] in ('flip', 'no-answers', 'drawn', 'commitment', 'batch', 'rehearsal', 'erase-kept-files')]
if valued: refuse('%s takes no value' % ', '.join(valued))
for o in ('report', 'ledger'):
    if o not in opts: continue
    if not opts[o]: refuse('--%s= names no file' % o)
    opts[o] = os.path.abspath(opts[o])  # one absolute path for the checks and for the writing; ".." is removed as text, so "link/../x" becomes "x" whatever `link` points to
    if os.path.lexists(opts[o]): refuse('--%s=%s: the file exists already, and nothing is overwritten: name a new file' % (o, opts[o]))
    found = others_can_change(os.path.dirname(opts[o]))  # any finding rejects the call, before anything is read or written
    if found: refuse('--%s=%s: its folder does not exist: make it first' % (o, printable(opts[o])) if found[1] == GONE else refusal('file of --' + o, opts[o], *found))
    opts[o] = os.path.join(os.path.realpath(os.path.dirname(opts[o])), os.path.basename(opts[o]))  # the directory that was checked, with every symbolic link in its path resolved
if 'report' in opts and opts['report'] == opts.get('ledger'): refuse('--report and --ledger name the same file')
if set(opts) - known: refuse('unknown option: %s. %s' % (', '.join('--' + o for o in sorted(set(opts) - known)), USAGE))
if opts.get('level') == '2' and not LEVEL2: refuse('level 2 is not in the tree (README.md, «What a YES does not cover», 3)')
if opts.get('level') not in levels or len(words) != 4 or words[3] not in ('1', '2'): refuse(USAGE)
tree, pubdir, world, field = words[0], words[1], words[2], int(words[3])
# docs/format/format_rules.py is imported from <tree>, below: only from the repository that this
# file is in: otherwise a relative or mistyped word would run a file of that name from another place.
if os.path.realpath(tree) != os.path.dirname(os.path.dirname(os.path.realpath(ME))): refuse('<tree> must be the repository that this audit.py is in: %s' % os.path.dirname(os.path.dirname(ME)))
tree = os.path.dirname(os.path.dirname(os.path.realpath(ME)))
commitment = state['commitment'] = 'commitment' in opts
if commitment and (opts['level'] != '1' or set(opts) - {'level', 'commitment', 'rehearsal', 'erase-kept-files', 'config', 'ivk', 'isk'}): refuse('--commitment goes with --level=1, --rehearsal or --erase-kept-files and, at most, --ivk=, --isk=, --config=')
batch = state['batch'] = 'batch' in opts
if batch and (opts['level'] != '1' or world != '-' or not {'ivk', 'isk', 'in', 'out'} <= set(opts) or set(opts) - {'level', 'batch', 'rehearsal', 'erase-kept-files', 'config', 'ivk', 'isk', 'in', 'out'}):
    refuse('--batch goes with --level=1, «-» for the world, --ivk=, --isk=, --in=, --out=, --rehearsal or --erase-kept-files and, at most, --config=')
if not batch and {'in', 'out'} & set(opts): refuse('--in= and --out= go with --batch')
places = {k: opts[k] for k in ('config', 'ivk', 'isk', 'in', 'out') if k in opts}
if not all(places.values()): refuse('a folder\'s name is empty: %s' % ', '.join('--%s=' % k for k, v in places.items() if not v))
if 'isk' in places and 'config' not in places and world == '-': places['config'] = os.path.join(places['isk'], 'config')  # the committer's own copy of the three public files
if world == '-' and not {'config', 'ivk', 'isk'} <= set(places): refuse('with «-» for the world, --ivk=<folder> and --isk=<folder> say where the commitment lies')
def lies(p):  # the path on disk of `p`, which is relative to <files>
    head, _, rest = p.partition('/')
    return os.path.join(places[head], rest) if head in places else os.path.join(world, p)
if ('rehearsal' in opts or 'erase-kept-files' in opts) and not (commitment or batch): refuse('--rehearsal and --erase-kept-files go with --commitment or --batch')
if (commitment or batch) and ('rehearsal' in opts) == ('erase-kept-files' in opts): refuse('--commitment and --batch want exactly one of --rehearsal (all the checks; nothing is destroyed or erased) and --erase-kept-files (the audit itself: it destroys or erases the kept files)')
rehearse = 'erase-kept-files' not in opts  # nothing is destroyed or erased unless the caller has asked for it by that word
# A slip in the call would end a batch's audit as NOT WHOLE, and with --erase-kept-files every
# verdict erases the kept file. So these are rejected here, before any file is read: a directory
# that does not exist (--isk= too, so that the message names it), one for --ivk= without key.bin or
# leaf/ (both as in audit_batch.sh), and a field other than 2, the only field that cp_prove has.
for what, p in [('pubtool\'s output folder ', pubdir)] + [('--%s=' % k, places[k]) for k in ('isk', 'config', 'ivk', 'in', 'out')] if batch else ():
    if not os.path.isdir(p): refuse('%s%s is not there, or is no folder (a mistyped name?).' % (what, printable(p)))
if batch and not (os.path.isfile(lies('ivk/key.bin')) and os.path.isdir(lies('ivk/leaf'))): refuse('--ivk=%s holds no key.bin, or no folder leaf: this is not a public folder of cp_commit\'s (a mistyped name?).' % printable(places['ivk']))
if batch and field != 2: refuse('the fourth word, the field\'s number, must be 2 for a batch: cp_prove has no other field (a mistyped number?).')
# Synthetic weights only, like the programs' test options (made_up() in prover/common/cp_files.h).
# <files> without isk/ has no secret key directory to protect.
isk_txt = lies('isk/isk.txt')
# isk.txt's line `seed`: the seed came from a test (`cp_commit -test-seed`) or from a file that may
# be published (-public-weights). Files made under it hide nothing, and each YES line says so. Under
# `test` they are not to be handed out; under `public` they may be.
OPEN_SEED = {'test': 'A TEST\'s SEED', 'public': 'A PUBLIC SEED'}
if os.path.isdir(lies('isk')) or commitment or batch or 'drawn' in opts:
    if not os.path.isfile(isk_txt): refuse('no ' + isk_txt)
    isk_text = open(isk_txt).read()  # read once: under --commitment and --batch the lines `weights`, `first`, `M` and `seed` all come from these bytes
    if [l for l in isk_text.split('\n') if l.startswith('weights ')] not in (['weights generated'], ['weights made-up-blocks']):
        refuse('%s has no line «weights generated» or «weights made-up-blocks». This tool runs on made-up weights only.' % isk_txt)
if commitment or batch:
    import kept
    sys.stdout.reconfigure(line_buffering=True)  # if the output's reader has exited, the next print fails, long before anything is destroyed
    print('the secret folder: %s' % os.path.realpath(lies('isk')))
    hard = kept.harden(); print('this process: ' + hard)
    if 'CORE FILES NOT SHUT OUT' in hard or 'STILL DUMPABLE' in hard: refuse('core files or dumps of this process cannot be shut out here: it would hold the seed and the kept bytes')
sys.path.insert(0, tree + '/docs/format')
sys.path.insert(0, ME)
import format_rules as R
import cut

_read = {}
def rd(p):  # reads each file once: parsing, comparisons and digests all see the same bytes
    if p not in _read: _read[p] = open(lies(p), 'rb').read()
    return _read[p]
if commitment and not (os.path.isfile(lies('ivk/key.bin')) and os.path.isdir(lies('ivk/leaf'))):
    print('audit.py: field %d: THE COMMITMENT: NOT WHOLE (ivk/key.bin or ivk/leaf/ is not there: the committer has not ended well). It counts as NO. Nothing was destroyed.' % field); sys.exit(4)
if batch:
    import batch as BA, field as FD, level1, level1_proof
    FD.lib()  # loaded now: if a module or libfield.so is missing, the audit stops here (status 7) before it looks at the kept file, and the batch can still be audited
    HEAD = 'audit.py: field %d: THE BATCH: ' % field
    box = kept.KeptBatch(lies('isk'))  # opens isk/audit/ once; its files are reached only through that descriptor
    st, why = box.state()
    if st == 'nothing': print(HEAD + 'NOTHING KEPT HERE (%s). Nothing was erased. Made without cp_prove\'s -keep-for-the-audit, or audited already: such a batch cannot be audited. Unless you have this batch\'s YES in your log, none of its responses may be handed out: make a new batch. If other files are in isk/audit/, they are not a batch\'s: a commitment\'s kept files are for audit_commitment.sh.' % why); sys.exit(5)

    def batch_end(word, why, more=()):
        """Prints the verdict `word` (YES, NO or NOT WHOLE), erases the kept file and exits.

        Order: one flushed line, the erasure, the lines of `more`, the verdict line. If the first
        line cannot be written, no one is reading, and nothing is erased (status 7). YES is printed
        only after the file is gone and isk/audit/ is empty; otherwise it becomes NOT WHOLE. In a
        rehearsal (--rehearsal) nothing is erased."""
        state['end'] = None
        box.wipe()
        leave = 'None of this batch\'s responses may be handed out.' if word != 'YES' else '' if text.get('seed') == 'test' else 'Compare the digests above with what you hand out. Keep these lines: nothing prints them a second time.'
        if rehearse:
            print(HEAD + ('REHEARSAL PASSED: %s. This is not a YES.' if word == 'YES' else word + ': %s.') % why + ' Nothing was erased (--rehearsal): isk/audit/batch.fresh is still there, and until an audit with --erase-kept-files has erased it hiding is void for this batch. ' + (leave if word != 'YES' else ''))
            sys.stdout.flush(); os._exit(6 if word == 'YES' else 1 if word == 'NO' else 4)
        try: print('the audit has come to its verdict; it now erases «isk/audit/batch.fresh»'); sys.stdout.flush()  # contains « », as a YES line does: output that cannot encode them fails here, before the erasure
        except Exception: os._exit(7)
        state['begun'] = True
        try: left, stopped = box.erase(state), None
        except Exception as e: left, stopped = [], str(e) if isinstance(e, kept.Stop) else type(e).__name__
        if not stopped and not left and state['destroyed'] != 1: stopped = 'nothing was erased, and isk/audit/ is empty'  # "erased" is printed only after a counted unlink; `erase` by itself does not lead here
        if stopped: gone = 'THE ERASURE STOPPED (%s): the batch\'s 32 bytes may still exist, in isk/audit/ or under another name. Hiding stays void for this batch, and cp_prove refuses the secret folder while anything is in isk/audit/. Tell the person in charge.' % stopped
        elif left: gone = '%s LEFT ALONE in isk/audit/: %s. cp_prove refuses the secret folder while anything is there. Tell the person in charge.' % (
            'batch.fresh is erased.' if state['destroyed'] else 'NOTHING WAS ERASED: batch.fresh is not a regular file of at most 32 bytes with one name that belongs to the caller, or it cannot be opened for writing. Hiding stays void for this batch while that file, or another name of it, exists.', kept.shown(left))
        else: gone = 'The kept file is erased: overwritten, synced, unlinked.'
        if word == 'YES' and (stopped or left): word, why, leave, more = 'NOT WHOLE', 'every check passed, but the erasure did not end as it should', 'None of this batch\'s responses may be handed out.', ()
        try:
            for l in more: print(l)
            print(HEAD + '%s: %s. %s %s' % (word, why, gone, leave)); sys.stdout.flush()
        except Exception: os._exit(7)  # no one read the verdict; if the file is gone, the next run prints NOTHING KEPT HERE
        os._exit({'YES': 0, 'NO': 1, 'NOT WHOLE': 4}[word])
    # From here on every ending goes through batch_end(), which erases the kept file, or leaves it
    # alone and says so in the verdict line (KeptBatch.erase() in kept.py). The exceptions are a
    # keyboard interrupt, output that cannot be written (both status 7) and a fatal signal: these
    # erase nothing, unless the erasure had begun.
    state['end'] = batch_end
    if st == 'not whole': batch_end('NOT WHOLE', why)
    kept_bytes = box.read()['batch']
    if not (os.path.isfile(lies('ivk/key.bin')) and os.path.isdir(lies('ivk/leaf'))): batch_end('NOT WHOLE', 'ivk/key.bin or ivk/leaf/ is not there')
conf = R.configuration_digest([rd('config/policy.txt'), rd('config/layout.txt'), rd('config/KL.bin')])  # docs/format/FORMAT.md, R4.9, which reports this order as the three programs' practice
C = cut.Cutter(R, field, conf, lambda n: open(os.path.join(pubdir, n), 'rb').read())
# level1.run() takes `W` and adds `W['ID']`, the commitment ID.
W = dict(C=C, R=R, field=field, world=world, at=lies, conf=conf, rd=rd, opts=opts, tree=tree)
W['key'] = C.key('key.bin', rd('ivk/key.bin'))
copies = sorted(int(n[:-4]) for n in os.listdir(lies('ivk/leaf')) if n.endswith('.bin') and n[:-4].isdigit())
W['entries'] = {k: C.entry('leaf/%d.bin' % k, rd('ivk/leaf/%d.bin' % k), W['key']['root'], k, W['key']['answers']) for k in copies}
names = sorted(n[:-6] for n in os.listdir(lies('out')) if n.endswith('.proof')) if os.path.isdir(lies('out')) else []
if 'proofs' in opts: names = [n for n in opts['proofs'].split(',') if n != '-']
if commitment: names = []
if batch:
    text = dict(l.split(' ', 1) for l in isk_text.splitlines()); live = list(range(int(text.get('first', 0)), int(text['M'])))
    if not live or not set(live) <= set(copies): batch_end('NOT WHOLE', 'ivk/leaf/ lacks the entry of a stored copy that is in use by isk.txt (its lines «first» and «M»)')
    lack = [k for k in live if not os.path.isfile(lies('isk/leaf/%d/fresh.bin' % k))]  # only the live leaf tables', as in `cp_prove` (`read_isk` in prover/prover/isk.cuh): a retired one's secret directory may be gone
    if lack: batch_end('NOT WHOLE', 'isk/leaf/<k>/fresh.bin is missing for %d of the %d stored copies in use: without it a copy\'s random values cannot be derived' % (len(lack), len(live)))
    word = {}
    for x in (l.split(' ') for l in rd('config/policy.txt').decode('ascii').split('\n')):
        if len(x) > 1: word.setdefault(x[0], x[1])  # a keyword's first line counts, as in `statement_words` of docs/format/format_rules.py
    try: slots, files, digest = BA.read(lies('in'), lies('out'), 1 << int(word['lt']), int(word['V']), int(word['slots']))
    except BA.Unfit as e: batch_end('NOT WHOLE', str(e))
    _read.update(files)
    names = [n for n, w, proved in slots if proved]
    # Nothing that is kept ties the directory of -in to the batch. If files were added, removed,
    # renamed or (a failed request's) changed since `cp_prove` ran and the slots now differ, the
    # batch's first key, from which its random values derive, differs too: unless BA.read() rejects
    # the directories, an honest batch gets a NO that cannot be told from a fault.
    print('the batch: %d slot(s): %d with a reply and a proof, %d failed (%d of these files are not well-formed requests); sha256 over the slots\' names and request files: %s' % (len(slots), len(names), len(slots) - len(names), sum(w is None for _, w, _ in slots), digest))
    if not names: batch_end('NOT WHOLE', 'every request failed: there is no proof to compare')
    W['batch'], W['slots'], W['copies'], W['live'] = kept_bytes, [(n, w) for n, w, _ in slots], (int(text['M']), live[0]), live

over = []  # published files that the audit does not read
for d, read in (('ivk', {'key.bin', 'leaf'}), ('ivk/leaf', {'%d.bin' % k for k in copies}), ('out', {n + e for n in names for e in ('.proof', '.y')})):
    if os.path.isdir(lies(d)) and not ((commitment or batch or 'proofs' in opts) and d == 'out'): over += [d + '/' + n for n in sorted(os.listdir(lies(d))) if n not in read]  # with --proofs the caller leaves files of out/ unread on purpose; --batch: `BA.read` has checked out/

if commitment:
    HEAD = 'audit.py: field %d: THE COMMITMENT: ' % field
    text = dict(l.split(' ', 1) for l in isk_text.splitlines()); live = list(range(int(text.get('first', 0)), int(text['M'])))
    if not live: refuse('%s names no live entry' % isk_txt)
    box = kept.Kept(lies('isk'), copies)  # opens isk/audit/ once; its files are reached only through that descriptor
    st, why = box.state()
    if st == 'whole' and copies != live: st, why = 'not whole', 'ivk/leaf/ does not hold exactly the entries %d to %d that isk.txt names' % (live[0], live[-1])
    if st == 'whole' and over: st, why = 'not whole', 'ivk/ holds %d file(s) that the audit does not read: what is audited must be what leaves' % len(over)
    if st == 'nothing': print(HEAD + 'NOTHING KEPT HERE (%s). %s' % (why, 'Made without -keep-for-the-audit, or the folder was removed by a hand: it cannot be audited.' if box.fd is None else 'After a YES: nothing to do. Before a first YES it counts as NO.')); sys.exit(5)
    if st == 'not whole': print(HEAD + 'NOT WHOLE (%s). It counts as NO. Nothing was destroyed.' % why); sys.exit(4)
    W['kept'] = box.read()
if batch and over: batch_end('NOT WHOLE', 'ivk/ holds %d file(s) that the audit does not read' % len(over))
W['proofs'] = {n: C.proof(n + '.proof', rd('out/%s.proof' % n), {k: e['root'] for k, e in W['entries'].items()}, rd('config/policy.txt'), rd('config/KL.bin'), rd('in/%s.req' % n), {k: e['answers'] for k, e in W['entries'].items()}) for n in names}
if batch and len(slots) > 1 << W['proofs'][names[0]]['depth']: batch_end('NOT WHOLE', 'the folder of -in holds more requests than a batch has slots')
sizes = {'key.bin': len(rd('ivk/key.bin')), **{'leaf/%d.bin' % k: len(rd('ivk/leaf/%d.bin' % k)) for k in copies}, **{n + '.proof': len(rd('out/%s.proof' % n)) for n in names}}

at = collections.Counter()
for r in C.recs:
    assert r.off == at[r.file], '%s: a gap or an overlap at byte %d' % (r.file, r.off)
    at[r.file] += r.n
assert dict(at) == sizes, 'a file is not booked whole'
print('LEVEL 0: %d files cut whole, %d bytes, %d records, %d trees; every chain replayed (every nonce holds), every opening ends in its root; no byte unbooked' % (len(sizes), sum(sizes.values()), len(C.recs), len(C.trees)))
print('files in ivk/, ivk/leaf/, out/ that the audit does not read: %d%s' % (len(over), ': ' + ', '.join(over[:8]) if over else ''))

cries = ['1 x (F) a file in a published folder that the audit does not read: %s' % p for p in over]
# (Y) The next token y (`reply` in the code) is sent unmasked, as (y, 0, 0) in the extension
# field E, in the round `REPLY` (docs/format/FORMAT.md, R14.8, R15.7); the token file
# (`out/<n>.y`) contains it in 4 bytes.
for n in names:
    reply = [r for r in C.recs if r.file == n + '.proof' and r.cls == 'reply']
    y = rd('out/%s.y' % n) if 'out/%s.y' % n in _read or os.path.isfile(lies('out/%s.y' % n)) else None
    if y is None or len(y) != 4 or len(reply) != 1 or rd('out/%s.proof' % n)[reply[0].off:reply[0].off + reply[0].n] != y + bytes(20): cries.append('1 x (Y) out/%s.y is not the reply that out/%s.proof sends' % (n, n))
unknown = collections.Counter(r.sub for r in C.recs if r.cls == 'unknown-scalar')
for n, c in unknown.items(): cries.append('%d x (N) a piece of a prover\'s message under a name that the audit does not know: «%s»' % (c, n))
if 'no-answers' in opts:
    print('NO ANSWERS: the keys are derived without the stored answers (%d answers are left out)' % ((len(W['key']['answers']) + sum(len(e['answers']) for e in W['entries'].values())) // 24))
    for t in [W['key']] + list(W['entries'].values()): t['answers'] = b''
if int(opts['level']) >= 1:
    import level1
    cries += level1.run(W)


def h4():
    """Check (H4): needs no secret, and runs at every level.

    A scalar that the protocol sends under a mask of its own should be uniform in E (192 bits in
    field 2), so among the elements of all the files none is 0 and no two are equal, except with
    negligible probability. The shared chain's records are the same bytes in every proof of one
    batch: that is not taken for a repeat.
    Detected: a zero mask where the masked value is 0 or repeats (a hiding sum over zero mask
    entries is 0); one proof's scalars repeated in another.
    Not detected: a mask that is zero or not applied, over a value that does not repeat; two
    batches whose shared chains are equal byte for byte (they count as one batch)."""
    data = {'key.bin': rd('ivk/key.bin'), **{'leaf/%d.bin' % k: rd('ivk/leaf/%d.bin' % k) for k in copies}, **{n + '.proof': rd('out/%s.proof' % n) for n in names}}
    batch = {}
    for r in C.recs:  # (the shared chain has no random bytes of its own in a proof: its first root tells the batches apart)
        if r.cls == 'root' and r.chain == 'shared' and r.file.endswith('.proof'): batch.setdefault(r.file, data[r.file][r.off:r.off + r.n])
    seen, zero, twice, total, nth = {}, collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for r in C.recs:
        if r.cls not in ('hiding-sum', 'sumcheck-coefficient', 'end-value', 'subproof-scalar', 'fold-coefficient', 'split-value', 'base-target', 'sample-answer', 'record-answer'): continue  # (an answer of a stored table's record is uniform if its pad is there: without pads a row that contains no weight has the answer 0, and equal rows of one table have equal answers)
        b = data[r.file][r.off:r.off + r.n]; nth[(r.file, r.chain, r.rnd)] += 1
        for i in range(0, len(b), 24):
            e = b[i:i + 24]; total[r.cls] += 1
            at = ('shared', batch[r.file], r.rnd, nth[(r.file, r.chain, r.rnd)], i) if r.chain == 'shared' and r.file in batch else (r.file, r.chain, r.rnd, r.off + i)
            if e == bytes(24): zero[r.cls] += 1
            elif seen.setdefault(e, at) != at: twice[r.cls] += 1
    print('(H4) scalars under masks of their own: %d elements; equal to 0: %s; standing at two places: %s' % (sum(total.values()), dict(zero) or 'none', dict(twice) or 'none'))
    return ['%d x (H4) %s: a scalar that should be uniform is 0' % (n, c) for c, n in sorted(zero.items())] + ['%d x (H4) %s: one value stands at two places' % (n, c) for c, n in sorted(twice.items())]
cries += h4()

kind = lambda f: 'key' if f == 'key.bin' else 'entry' if f.startswith('leaf/') else 'proof'
tab = collections.OrderedDict()
for r in C.recs:
    k = (kind(r.file), r.cls + ('/' + r.sub if r.sub and r.cls in ('main-symbol', 'mask-symbol', 'subproof-scalar', 'sample-answer', 'base-word', 'hiding-sum', 'unknown-scalar') else ''), r.level)
    t = tab.setdefault(k, [0, 0, 0]); t[0] += 1; t[1] += r.n; t[2] += r.bad
lines = ['file\tclass\tlevel\trecords\tbytes\twords of 8 bytes\tNOT AS DERIVED (records)']
for k in sorted(tab): lines.append('%s\t%s\t%d\t%d\t%d\t%.1f\t%d' % (k + (tab[k][0], tab[k][1], tab[k][1] / 8, tab[k][2])))
tot = collections.Counter()
for k, t in tab.items(): tot[k[2]] += t[1]
lines.append('ALL\t\t\t%d\t%d\t%.1f\t%d' % (len(C.recs), sum(sizes.values()), sum(sizes.values()) / 8, sum(t[2] for t in tab.values())))
lines.append('by level: ' + '; '.join('level %d: %d bytes (%.2f %%)' % (l, b, 100 * b / sum(sizes.values())) for l, b in sorted(tot.items())))
print('\n'.join(lines))
by_chain = collections.Counter()
for r in C.recs: by_chain[(kind(r.file), r.chain or '-', r.cls + ('/' + r.sub if r.sub and r.cls in ('main-symbol', 'mask-symbol', 'sample-answer', 'base-word') else ''), r.level)] += r.n
if 'report' in opts:
    with open(opts['report'], 'x') as f:
        f.write('\n'.join(lines) + '\n\n== bytes by file, chain, class, level\n' + '\n'.join('%s\t%s\t%s\t%d\t%d' % (k + (v,)) for k, v in sorted(by_chain.items())) + '\n\n== openings\n' + '\n'.join(C.log) + '\n\n== notes\n' + '\n'.join(W.get('notes', [])) + '\n\n== cries\n' + '\n'.join(cries) + '\n')
most = len(cries) if commitment or batch else 40  # --commitment and --batch allow no --report, so every kind is printed
for c in cries[:most]: print('CRY: ' + c)
if len(cries) > most: print('CRY: .. and %d kinds more%s' % (len(cries) - 40, ' (all in the report)' if 'report' in opts else ' (--report=<file> has all)'))
if commitment:
    box.wipe()
    if cries: print(HEAD + 'NO: %d kinds of failed checks, listed above. The kept files stand. Nothing of this commitment may leave.' % len(cries)); sys.exit(1)
    reached = 'key file and %d entries: every level-1 check passed (NOT checked: pads inside tables of real values, %sthe stored tables\' columns, unopened leaves of trees that hold real values: README.md, «What a YES does not cover»)' % (len(copies), 'the pads of the stored tables\' records, ' if W['key']['answers'] else '')
    last = 'A TEST\'s SEED: nothing is hidden in this commitment, and it is not to be handed out.' if text.get('seed') == 'test' else ('A PUBLIC SEED: nothing is hidden in this commitment: it is for weights that are public. ' if text.get('seed') == 'public' else '') + 'Compare the digests above with what you hand out. Keep these lines: nothing prints them a second time.'
    if rehearse: print(HEAD + 'REHEARSAL PASSED: %s. Nothing destroyed (--rehearsal); this is not a YES: for that, run it with --erase-kept-files.' % reached); sys.exit(6)
    try: print('every check has passed: the kept files in «isk/audit/» are destroyed now'); sys.stdout.flush()  # flushed now: if the output's reader has exited, this fails and nothing is destroyed; it contains « », as the YES line does: output that cannot encode them fails here, before the destruction
    except Exception: os._exit(7)
    state['begun'] = True
    try: box.destroy(state)
    except Exception as e: print(HEAD + 'NOT WHOLE: the destruction stopped (%s; %d of %d destroyed, one more may be overwritten). It counts as NO.' % (e if isinstance(e, kept.Stop) else type(e).__name__, state['destroyed'], len(copies) + 1)); sys.exit(4)
    try:
        for p in ['ivk/key.bin'] + ['ivk/leaf/%d.bin' % k for k in copies]: print('audited: %s sha256 %s' % (p, hashlib.sha256(rd(p)).hexdigest()))
        print('audited: the ID %s' % W['ID'])
        print(HEAD + 'YES: %s; the %d kept files are destroyed. %s' % (reached, state['destroyed'], last)); sys.stdout.flush()
    except Exception: os._exit(7)  # the files are gone and no one read the YES: the next run prints NOTHING KEPT HERE, which before a first YES counts as NO
    os._exit(0)
if batch:
    if cries: batch_end('NO', '%d kind(s) of failed checks, listed above' % len(cries))
    apart, retired = W.get('siblings not compared', 0), W.get('salts of retired copies', 0)
    batch_end('YES', '%d proof(s) of a batch of %d slot(s): every level-1 check passed%s (NOT checked: pads inside tables of real values, %sthe stored copies\' columns, salts of leaves that no proof opens, the key file and the entries but for the stored tables\' salts: README.md, «What a batch\'s YES does not cover»)%s' % (
        len(names), len(slots), ('; %d sibling(s) in the tree that ties the batch together (the binding tree) were not compared: a failed request\'s leaf of that tree is in no file' % apart if apart else '') + ('; %d salts of retired copies\' tables, in their entries, were not compared: their fresh.bin is gone' % retired if retired else ''), 'the pads of the stored tables\' records, ' if W['key']['answers'] else '', '. %s: nothing is hidden in this batch' % OPEN_SEED[text['seed']] if text.get('seed') in OPEN_SEED else ''),
        ['audited: out/%s%s sha256 %s' % (n, e, hashlib.sha256(rd('out/%s%s' % (n, e))).hexdigest()) for n in names for e in ('.y', '.proof')])
own = hashlib.sha256(b''.join(open(os.path.join(ME, n), 'rb').read() for n in sorted(os.listdir(ME)) if n.endswith(('.py', '.cpp')))).hexdigest()[:16]
print('options: %s; sha256 of this folder\'s .py and .cpp (16 digits): %s%s' % (' '.join('--%s=%s' % kv for kv in sorted(opts.items()) if kv[0] != 'report'), own, '; of M.0: ' + W['M sum'] if 'M sum' in W else ''))
public = os.path.isfile(isk_txt) and 'seed public' in open(isk_txt).read().split('\n')
print('audit.py: field %d: LEVEL %s over 1 key, %d entries, %d proofs%s: %s' % (field, opts['level'], len(copies), len(names), (' (WITH BITS TURNED: --flip)' if 'flip' in opts else '') + (' (KEYS DERIVED WITHOUT THE ANSWERS: --no-answers)' if 'no-answers' in opts else '') + ('' if 'drawn' not in opts else ' AS UNDER A DRAWN SEED (--drawn): THE STORED TABLES\' COINS ALONE ARE HELD' if public else ' UNDER A DRAWN SEED: THE STORED TABLES\' COINS ALONE ARE HELD') + (' (A PUBLIC SEED: NOTHING IS HIDDEN IN THESE FILES)' if public else ''), 'NOTHING CRIED' if not cries else '%d KINDS OF CRIES (listed above)' % len(cries)))
sys.exit(1 if cries else 0)
