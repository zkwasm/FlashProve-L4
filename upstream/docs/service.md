# The prover as a service

**In short.** Run as a service, `cp_prove` watches one directory, the [spool
directory](glossary.md#spool-directory). Your program, the [front end](glossary.md#front-end), drops
request files there and picks up a response for each one, either a proof or a "failed" marker.
`cp_prove` runs no verifier: it releases a batch's responses once all of them are staged, and
nothing has then checked that a proof verifies. That check is the front end's: the vLLM integration
that ships hands on no proof, and nothing of its response, before `cp_verify` has passed every proof
of its batch (narrow exceptions, none open to a client:
[integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)); `driver/measure_rate.py`
reports no rate, and an example script shows no answer, before `cp_verify` has accepted the proofs;
and a front end of your own must do the like. `cp_prove` does little to protect the spool directory
either, so your program must take [nine steps](#what-your-program-must-do) itself. Above all, keep
clients out of the directory, check each response's tag and its proof, and stop after any exit code
other than 0.

This document is for anyone writing a front end. If you run the service, read [operator's
duties](operators-duties.md). To check a proof, read [verifying](verifying.md). What is not hidden
is told in [what is not hidden](what-is-not-hidden.md). A row ID such as R18.4 names a row of [the table of
all rows](limits-all-rows.tsv).

## How a request becomes a response

1. The front end writes a [request](glossary.md#request) file into `in/`.
2. `cp_prove` moves the oldest queued requests, up to `-batch` of them, into `work/` and proves them
   together as one [batch](glossary.md#batch). It does not wait for a batch to fill. If `in/` holds
   more than 100,000 requests, those that it takes need not be the oldest.
3. It writes each [response](glossary.md#response) under `out/.tmp/`.
4. Once the batch's responses are all written there, `cp_prove` puts them into `out/`. It runs no
   verifier on them: that is your program's to do ([step 5](#what-your-program-must-do)).

A request that reaches `work/` gets a response, with one exception. A file in `work/` that can no
longer be read, or whose contents no longer hash to its name, is deleted without a response.

If `cp_prove` exits or is killed before it responds, the next `cp_prove` that starts on the
directory marks the request "failed". A request file leaves `work/` once its response is on disk.

## What your program must do

1. **Keep [clients](glossary.md#client) out of the spool directory.** Only the front end reads and
   writes there, under the same account as `cp_prove`.
2. **Check and submit each request** ([A request file](#a-request-file)). Record the request ID and
   the tag key alongside the client's name, and never write the key to a log. Keep the request's
   words too, without the nonce: step 5's `cp_verify` takes them, and `cp_prove` removes its own
   copy.
3. **Limit the queue.** Nothing limits the number of files in `in/` or `out/`, so cap how many
   requests each client may have waiting and how many responses it may leave unfetched.
4. **Wait for `out/<id>.proof` or `out/<id>.failed`.** Some malformed requests never get a response,
   so set a timeout of your own.
5. **Check the response's tag, and then the proofs of its batch,** before you use the response. The
   tag: [The tag](#the-tag). The proofs: before you give out any response of a batch, a `failed`
   too, check every proof of that batch with a `cp_verify` that you built and keep where the
   service's account cannot write ([operator's duties](operators-duties.md#the-front-end)); the
   files of one batch have one modification time ([A response](#a-response)). `cp_prove` has checked
   none of them. A proof that `cp_verify` does not accept goes to nobody, and your program then
   stops as it does after an exit code other than 0 (step 9).
6. **Give the client only the contents of `.y` and `.proof`,** or tell it that the request failed.
   The tag is for the front end only, and the files' timestamps, number and sizes reveal additional
   information ([what is not hidden](what-is-not-hidden.md#what-a-client-learns)).
7. **Delete the response** once the client has it ([A response](#a-response)).
8. **Do not rely on `out/` while `cp_prove` is starting** ([What `cp_prove` does at
   startup](#what-a-start-does-to-the-spool-directory)).
9. **Stop after any exit code other than 0** ([How the service ends](#how-the-service-ends)).

`cp_prove` leaves no record of its exit code, and does not tell the front end when it starts. So for
steps 8 and 9, whatever launches `cp_prove` must report both to your program. This repository has no
complete front end, only an example ([The example front end](#the-example-front-end)) and some
[Python helpers](#python-helpers). [Operator's duties](operators-duties.md#the-front-end) lists what
else a front end owes the operator. The five rules that spell out [duty
11](limits.md#the-conditions) stand in [what a front end must
do](what-is-not-hidden.md#front-end-rules).

## The files

`<id>` is the request ID, 64 lower-case hex digits. On this page "ID" by itself means the request
ID, and the [commitment ID](glossary.md#commitment-id) is always spelled out.

| Path in the spool directory | Written by | What it is |
|---|---|---|
| `in/.tmp/` | the front end | a partly written request |
| `in/<id>.req` | the front end | a queued request |
| `work/<id>.req` | `cp_prove` | a request taken into a batch |
| `out/.tmp/<id>/` | `cp_prove` | a response that is not released yet; do not read or list it |
| `out/<id>.y` | `cp_prove` | the [token file](glossary.md#token-file): one 32-bit word. If the request's range ends at `e` = T, it holds the [next token](glossary.md#next-token), which the proof covers. If `e` is below T it is 0, and `cp_verify` rejects any other value there. |
| `out/<id>.tag` | `cp_prove` | 32 bytes, for the front end only ([The tag](#the-tag)) |
| `out/<id>.proof` | `cp_prove` | the [proof file](glossary.md#proof-file) |
| `out/<id>.failed` | `cp_prove` | an empty file, meaning that this request has no proof. It is the same whatever the cause. |
| `stop` | you | an empty file that asks the service to stop |
| `mark.txt` | `cp_prove`, or a front end that comes first | three lines that mark the directory as a spool directory ([The mark file](#the-mark-file)) |

`cp_prove` creates whichever of the five directories (`in/`, `in/.tmp/`, `work/`, `out/` and
`out/.tmp/`) are missing, with mode 0700.

## A request file

A request file is a sequence of little-endian 32-bit words, followed by 32 random bytes. A proof's
first hash takes the words alone, not these 32 bytes.

| Part | Size | Content |
|---|---|---|
| tokens | T words | the token IDs, each below the vocabulary size. T is set in the [parameter file](glossary.md#parameter-file). |
| `r`, `e` | 2 words | the range of positions to check, with `r < e <= T`. For each position t from `r` to `e - 1`, the proof covers token t + 1 ([how to choose the range](verifying.md#the-request-and-the-token-file)). |
| nonce | 32 bytes | fresh random bytes. The tag key is derived from them. |

The request's ID is the SHA-256 of the prefix `p258-spool-req-1` followed by the file's bytes.
`cp_verify` takes the same file without the nonce: given a spool's file as it lies, it ends
`NO: statement-range` (read in the code).

To submit a request:

1. Write the file under `in/.tmp/` and sync it.
2. Link it to `in/<id>.req` with a call that fails if the name exists.
3. Delete the temporary name.

`cp_prove` does not prove every file in `in/`:

| The file | What happens |
|---|---|
| its name is not `<64 lower-case hex digits>.req`, or it is not a regular file | It stays, and every scan reads it again, so keep `in/` for requests only. |
| its length is wrong for a request | It stays in `in/` and never gets a response, so check the length yourself. It does not count as a waiting request, so with `-end-when-empty` the service exits although it is there. |
| its name and length are a request's, but `cp_prove` is not allowed to read it (`Permission denied`) | From reading the code, it stays in `in/` and gets no response as long as each scan's try to open it is refused. A file that the service claims and then cannot read is removed, unanswered (seen once, in a test with two accounts), and a production build names it nowhere. So keep a copy of each request outside the spool directory. If no response comes and the file is gone from `in/` and from `work/`, submit the same file again, under the same name. While it still lies in `in/`, do not submit it a second time: mend its mode ([what to do](faults.md#unreadable-request)). `cp_prove` prints `p258 alarm unreadable-request` at most once and keeps running ([what to do](faults.md#unreadable-request)). |
| its name is not the hash of its bytes | It is deleted, and nothing ever appears in `out/`. |
| a token is not below the vocabulary size, or `r < e <= T` does not hold | The request is marked "failed". |
| its ID already has a response, or is being served | It is deleted, with no new response. |

To submit a request from a shell, set `S` to the spool directory's absolute path and run:

```sh
python3 driver/spool.py lay "${S:?}" REQUEST_FILE
```

`REQUEST_FILE` holds the tokens, `r` and `e` ([how to build
it](verifying.md#the-request-and-the-token-file)). The command adds the nonce, submits the request
and prints its ID. The spool directory must already exist. If it does not, the command prints an
error, exits with code 1 and creates nothing. If the spool directory lies below a directory that
another account may be able to change, the command names that directory, exits with code 2 and
submits nothing. Do not use this command for clients' requests: it does not keep the tag key, so the
response's tag cannot be checked, and it does not check the file's length or its tokens.

<a id="what-a-start-does-to-the-spool-directory"></a>
## What `cp_prove` does at startup

At startup `cp_prove` does the following, in order (particulars: rows R24, R18, R18.1, R13.1.1,
R13.1.2 and R36):

1. It checks the path that `-spool` names, and then the spool directory. It does not start, and
   touches nothing, if the path does not begin with `/`, or if any part of it is a symbolic link.
   Each directory along the path must belong to root or to `cp_prove`'s own account, and must not be
   writable by its group or by others, unless it has the sticky bit set, as `/tmp` has. The usual
   cause of a rejection is a directory with mode 0775, which `mkdir` creates under umask 002. After
   a rejection, pick a location with no symbolic link in its path, and give the path in full or as
   `"$PWD/..."`. If every link on the way is your own or root's, you can give the directory's real,
   absolute path instead, as the message says. If one belongs to another account, do not go by where
   it leads now. The spool directory itself must be owned by `cp_prove`'s own account and have no
   permission for group or others. `cp_prove` checks the path that `-isk` names, and the secret key
   directory, in the same way. From reading the code, it then locks that directory until it exits
   ([what the lock misses](operators-duties.md#stopping-and-starting): a copy of the directory,
   another machine that shares it, and another route to it on the same machine). Do not copy or move a secret key
   directory to get past a rejection.
2. It locks the directory. Only one process serves a spool directory at a time, so a second one does
   not start.
3. It looks for `mark.txt` ([The mark file](#the-mark-file)). It does not start, and touches nothing
   that is there, unless the directory has a well-formed `mark.txt` of the kind `spool` or is empty.
   It does not look at the commitment ID in the file. Into an empty directory it writes the file
   first. A directory still counts as empty if it holds only what a front end may put there before
   the first start: `in/` with nothing in it but `.tmp/` and request files, an empty `out/`, and an
   empty `stop` file. `in/`, `in/.tmp/` and `out/` must then be yours, with mode 0700.
4. It creates whichever of the five subdirectories are missing, and checks each one as it checked
   the spool directory. If you give `-spool` an empty directory of yours by mistake, `cp_prove`
   accepts it: `mark.txt` and the five subdirectories are created there, and they stay.
5. It looks for unexpected names. It does not start, and does not delete anything, unless every
   entry in `work/` is a regular file named `<64 lower-case hex digits>.req`, with the length of a
   request under this configuration. Nor does it start, or delete anything, if `out/.tmp/` holds a
   name the service never leaves there, or if `stop` is not an empty file. What an earlier process
   staged and never released then stays on the disk, until a later start gets through or [`cp_prove
   -clear`](faults.md#clear) clears it. With an unexpected name at the top of `out/.tmp/`, `-clear`
   too clears nothing until you have moved that entry out. These checks see only the top level of
   `work/` and `out/.tmp/`, and pass over the name `.tmp` there. If something that `cp_prove` did
   not stage lies deeper, inside a staged response's own directory, the start fails only after an
   empty `stop` and the staged responses that could be deleted are gone. The foreign file itself
   stays. After any of these rejections the subdirectories that step 4 has just created stay too
   (rows R18.1 and R36).
6. It deletes the empty `stop` file, if one is there. It then clears `out/.tmp/` by overwriting each
   staged response file with zeros and deleting it. In a marked directory it goes only by name: any
   entry with a name that `cp_prove` itself creates there counts as staged. It leaves every other
   entry untouched, and also a staged file that has a second hard link or that its own account does
   not own. If anything is left, it does not start. [`cp_prove -clear`](faults.md#clear) removes the
   same entries without a start. **Warning:** `cp_prove` knows no commitment ID, with `-clear` or
   without, so it accepts any well-formed `mark.txt`, and in a marked directory it decides what to
   delete only by name. So run it only on a directory that you passed to `cp_prove` as `-spool`, or
   as `-out` for a single batch.
7. For each request that an earlier process left in `work/`, it deletes any response in `out/`. The
   "failed" marker that replaces it appears only later, when serving begins. There is one exception:
   if a file in `work/` does not hash to its name, `cp_prove` deletes only that file and leaves a
   response with that ID in place.

`cp_prove` takes these seven steps before the rest of its setup, which takes much longer, so the
deletions happen even if startup then fails. A response you saw before the start may be gone
afterwards, with "failed" in its place.

**Under `sudo`.** Anything a run under `sudo` creates belongs to root, so use one account for the
spool directory, `cp_prove` and the front end. If you use `sudo` at all, run all of them with
`sudo`, or none of them ([Under `sudo`](operators-duties.md#under-sudo)). If you run `cp_prove`
under `sudo` even once, a later run without `sudo` is rejected, because root now owns what is inside
([what to do](faults.md#a-closed-directory-of-your-own); row R36). The spool directory itself may
belong to the account that ran `sudo`, with mode 0700. You then trust that account as much as root,
because the directory's owner can rename or replace `in/` while a request is being written. This
comes from reading the code. We have not run `cp_prove` itself under a real `sudo` (row R28 tells
six tests of its routines under one, through host programs, two of them with a bad outcome).

**Use one spool directory for each commitment, and keep them apart yourself.** `cp_prove` knows no
commitment ID, so it cannot tell one commitment's spool directory from another's: it starts on any
directory that has a well-formed `mark.txt`, or that counts as empty. If you give `-spool` the
directory of another commitment, requests of the right length are proved under the commitment whose
secret key directory you named with `-isk`. They use up part of that commitment, and the proofs fail
only when a verifier checks them against the commitment ID that the client expects (row R13.1.1 in
[the table of all rows](limits-all-rows.tsv)). A start there also clears what the other service had
staged, and answers "failed" to the requests that it had in `work/`. `cp_prove -clear` likewise
clears any marked directory. A copy or a restored backup of a spool directory has the same
`mark.txt`. Nor can `cp_prove` tell two spool directories of one commitment apart: if you give
`-spool` the wrong one of the two while no `cp_prove` runs on that secret key directory, it carries
out all its start-up steps there.

<a id="the-mark-file"></a>
### The mark file

`mark.txt`, at the top of a spool directory, has three lines, 99 bytes in all:

```text
cp-folder 1
kind spool
commitment COMMITMENT_ID
```

`COMMITMENT_ID` is 64 lower-case hex digits. `cp_prove` knows no commitment ID: where it writes the
file, the digits are 64 zeros, and it accepts any well-formed file, whatever its digits. It writes
the file at its first start on a directory that has none and counts as empty ([step
3](#what-a-start-does-to-the-spool-directory)). A front end that knows the [commitment
ID](glossary.md#commitment-id), such as the example front end, may write the file with that ID into
an empty directory before that, for its own checks. Your front end need not write it. No program
changes or deletes the file afterwards. From reading the code, `cp_prove` creates it with mode 0600.
So a `mark.txt` that a run under `sudo` wrote belongs to root, and a later run or a front end
without `sudo` cannot read it ([Under `sudo`](operators-duties.md#under-sudo)).

The file says that a directory is a spool directory, so that `cp_prove` deletes nothing in a
directory of another kind. It guards against that mistake alone, not against a mix-up of two
commitments, and not against someone who can write in the directory. For what to do if `cp_prove`
rejects a directory because of the file, see [operator's duties](operators-duties.md#the-mark-file).

## A response

A response is a set of files in `out/`, with a tag next to them ([The tag](#the-tag)). `cp_prove`
puts the files in place in a fixed order:

- for a proof: `<id>.y`, `<id>.tag`, then `<id>.proof`;
- for a failure: `<id>.tag`, then `<id>.failed`.

So a response is complete once `.proof` or `.failed` exists. `cp_prove` renames `.y`, `.proof` and
`.failed` into place. It creates the tag in `out/` and then writes it, so for a moment a tag can be
shorter than 32 bytes. All the files in a batch have the same modification time, the moment they
were released. `cp_prove` names them one after another, and no file says that a batch is whole: a
look into `out/` in between sees part of it. It removes the first of a batch's requests from `work/`
only when all the batch's files have their names and are on disk, so when `work/<id>.req` goes, the
whole batch of that ID is in `out/`. Two exceptions: a file in `work/` that is not the request that
its name says, or that cannot be read, is removed with nothing in `out/`; and if `cp_prove` is
killed, or the power fails, while it takes a batch's requests out of `work/`, the next start
withdraws the responses whose requests still lie there and answers them "failed", so that the batch
is no longer whole (from reading the code; no test that ships tries it). One release can also name a
"failed" for a request that got no place in a batch, with the same time.

**A complete response can still disappear.** `out/` can change at the next startup, so treat the
first complete response you read as the outcome for that ID: record it, and ignore later changes.
While `work/<id>.req` exists, the next startup deletes the response, `.y` first, and marks the
request "failed" instead. Once `work/<id>.req` is gone, no startup deletes the response. `cp_prove`
syncs that deletion to disk.

**An existing file in `out/` is not replaced.** If a file with a response's name is already there,
`cp_prove` silently discards its own file, so you would read the older one. Only the tag check is
sure to catch this. From reading the code, on a file system without the call that renames without
replacing, `cp_prove` looks whether the name is free and then renames, so a file that appears there
between the two steps is replaced.

**Sending a request again.** Every batch uses up part of the commitment, so while the service is
running you may resubmit a request marked "failed" once, but not in a loop. Give it a new nonce,
which also gives it a new ID ([Every day](operators-duties.md#every-day)).

**Deleting a response.** `out/` holds the only copy, and `cp_prove` never deletes a response just
because it is old. Delete the files once the client confirms receipt, or when it has not fetched
them within a time limit the operator sets. Delete `.proof` or `.failed` first, then `.tag`, then
`.y`. Build each name from the 64 hex digits of that one ID. Do not use a pattern or a listing, and
do not touch anything outside `out/`. `driver/spool.py` has no function for this.

### The tag

The tag shows the front end that a response belongs to the ID in its name. It does not show that the
proof is valid, and it does not protect against the service account or against anyone who could read
the request file.

| Value | How it is computed |
|---|---|
| key | HMAC-SHA-256, keyed with the nonce, of the prefix `p258-spool-tag-1` |
| tag of a proof | HMAC-SHA-256, keyed with the key, of the ID's 32 raw bytes, the lengths of `.y` and `.proof` as little-endian 64-bit words, the bytes of `.y`, and the bytes of `.proof` |
| tag of "failed" | HMAC-SHA-256, keyed with the key, of the ID's 32 raw bytes, 16 zero bytes, and the six ASCII bytes `failed` |

## How the service ends

To stop the service, create an empty file named `stop`
([how](operators-duties.md#stopping-and-starting)). `cp_prove` checks for it whenever it is about to
take a batch, and if the file is there, it finishes what it has already taken, deletes the file and
exits. Any entry named `stop` ends the service in this way, but `cp_prove` deletes only an empty
regular file. Anything else is left in place, and `cp_prove` does not start on the directory until
it is gone. Requests in `in/` stay there for the next process to serve.

`out/.tmp/` belongs only to `cp_prove`. As the last step of its memory wipe at exit, also at a
normal stop, `cp_prove` clears it, by the names that it uses there itself. If an entry of another
name is at the top of `out/.tmp/`, the exit counts as not fully wiped (after an otherwise good run:
exit code 1, `not-all-wiped`), and the next start is rejected ([an entry listed as
`LEFT`](faults.md#clear)). In the directory `out/.tmp/<id>/` of a response that is still staged,
this step deletes every entry that is not a directory, whatever its name, and prints nothing about
it. None of it was seen in a run of `cp_prove` (rows H26.1 and R18.4).

**What it prints.** `cp_prove` first prints lines on its settings. It prints no accounting of
soundness errors: `cp_verify -plan accounting`, which you run yourself, does ([operator's
duties](operators-duties.md#from-nothing-to-the-first-proof)). No line that it prints in a run says
that its responses are unchecked; only its help text (`-h`) does. Just before the [production
build](glossary.md#production-build) reads its first secret, the commitment's seed, it prints a line
that begins `p258 closed`. From then on it prints only lines from a fixed list ([the
list](../prover/docs/closed_front.md)). Even these depend on the requests, and perhaps on hidden
values. All these lines are for the operator, so do not pass any of them, or the exit code, on to a
client. One of the fixed lines is `batch <n> leaf <k> n <requests>`, printed once per released
batch. It does not say whether any request failed, and it can appear after the files do. The last
line gives the exit code and, for most exits with a code other than 0, a name for the cause.

| Exit code | Last line | What it means |
|---|---|---|
| 0 | `p258 end 0` | The service stopped on request and released every response that was due. All of them may be "failed", so exit code 0 does not mean that any request was proved. Nor does it mean that a proof verifies: `cp_prove` runs no verifier. |
| 1 | `p258 end 1`, usually with a name | Any other exit: a rejected call, a machine fault, a crash, a signal that asks it to exit, a full [leaf table](glossary.md#leaf-table), or a memory wipe that was not completed. A few self-checks lead to this exit code too, two of them with a bare `p258 end 1`. After a crash memory is not wiped. |
| 9 | `p258 end 9` with a name that begins `own-test-` | A self-check on secret random values or keys failed. From reading the code, one name, `own-test-coin-library`, can also mean that the operating system gave no memory page that a fork wipes: then no self-check has failed ([faults](faults.md#what-you-see-and-what-to-do)). Responses not yet in `out/` are not released, with three exceptions (rows C3 and HW-583), and the next start marks those requests "failed". This exit has been seen only in a test build with an injected fault, in small single calls of one request, never in a service or in a production build, so this description comes from reading the code. The self-checks cover only some of the random values (row LIM-67). |
| that of the signal | none | `KILL` stopped it. Unreleased files can remain under `out/.tmp/`, as they can after a power failure. They have not been verified and can hold unmasked values, so do not give them to anyone ([duty 6](limits.md#the-conditions)). |

[When something goes wrong](faults.md#what-you-see-and-what-to-do) says what each exit means for the
operator, and what a person should do next ([every
name](../prover/docs/ends-in-short.md#the-names-in-the-last-line)).

**After any exit code other than 0, your program must stop.** A new `cp_prove` does not know how the
last one ended and would go on serving, so a person must decide whether it may start ([the two exits
that need no hold](faults.md#always-first), each with its conditions). The same holds after a proof
that your own run of `cp_verify` did not accept: `cp_prove` does not learn of it and goes on.

- Do not submit anything.
- Do not deliver anything more, not even a complete response from an earlier batch, because the
  fault may have been present before it was noticed ([limits](limits.md#a-lost-mask)).
- Do not let a program restart the service, especially after exit code 9.
- Do not delete anything in the spool directory, and do not switch to a new directory. A next start
  on the same directory whose requests have the same length as those left in `work/` cleans up what
  is staged ([What `cp_prove` does at startup](#what-a-start-does-to-the-spool-directory)). It
  cannot stop a disk or a file system, such as a copy-on-write one, from keeping the old contents of
  overwritten blocks. It does not remove a released response whose request has left `work/`: one
  that your own `cp_verify` did not accept, and the rest of its batch, then stay in `out/` under
  their final names, so before a person lets the service start again, make sure that your program
  can never hand them out. If no service is started there again, then before you give up the
  directory, run `cp_prove -clear` on it, and on no other ([how to use it](faults.md#clear)).

No test kills or signals `cp_prove` on purpose at the [default
setting](glossary.md#default-setting), which asks for [proof of work](glossary.md#proof-of-work)
([what no test covers](limits-further.md#what-no-test-covers)).

## Python helpers

`driver/spool.py` builds a request, which is the client's part, and handles the spool directory from
the front end's side. Its functions cover part of what a front end does there, but none of them
checks a request, and none reads or writes `mark.txt`. `driver/measure_rate.py` uses these functions
to measure a service's rate of proving; it is no front end
([driver/README.md](../driver/README.md#the-rate)).

| Function | What it does |
|---|---|
| `private_folders(spool, ...)` | creates the spool directory and the directories that you name below it, each nested in the previous one, if they are missing. It checks that each existing one is a directory rather than a symlink, belongs to you, and gives group and others no permissions. |
| `lay(spool, request, with_key=True)` | submits a request, and returns its ID and the tag key |
| `answered(spool)` | returns the IDs that have a response |
| `wait_for(spool, ids, limit_s)` | waits until all of these IDs have a response, or time runs out |
| `read_member(spool, id, key=key)` | reads one response, checks its tag, and returns each file's SHA-256 and a list of errors, `faults` |
| `stop(spool)` | creates the `stop` file |

`private_folders(spool)` looks only at the spool directory, so before you read responses call
`private_folders(spool, "out")`, which looks at `out/` too. The function also creates any missing
directory above the spool directory, so a mistyped path leaves new directories behind.

Call `read_member` only for an ID that `answered` has returned; until then, `faults` is not empty.
Whenever `faults` is not empty, do not pass the response on to anyone. No function of
`driver/spool.py` runs a verifier: a response that `read_member` returns without faults can still be
one that `cp_verify` rejects (step 5 of [What your program must do](#what-your-program-must-do)).
`read_member` opens each file once, so if you later read a file by its path, check what you read
against the SHA-256 it returned. Without a key, it checks only the tag file's size, owner and mode,
so the response could belong to a different request.

**The functions in `driver/spool.py` trust the path that you give them.** In that file, only the
`lay` command checks the directories above the spool directory. An account that can write to a
directory on that path can put another directory in its place ([Shared
machines](operators-duties.md#shared-machines)).

## The example front end

`integrations/proof_middleware.py` is an example front end, written as middleware for a vLLM server
([integrations/vllm/README.md](../integrations/vllm/README.md)). It meets only some of a front end's
duties ([limits](limits.md#the-vllm-integration)). It does not meet [duty
11](limits.md#the-conditions). [integrations/LIMITS.md](../integrations/LIMITS.md) lists its limits.

- **It says nothing of a response, a `failed` one too, until `cp_verify` has passed every proof of
  its batch that belongs to a job** (narrow exceptions, none open to a client:
  [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions); step 5 of [What your
  program must do](#what-your-program-must-do)). The check starts when a client first asks for a job
  of the batch, so a batch that nobody asks for is not checked. It has two ends. At a rejection the
  front end writes a line to its log and puts the policy on hold (where it can write nothing, only
  the running server's memory stops the policy: read
  [integrations/LIMITS.md](../integrations/LIMITS.md) before you stop or restart it), overwrites
  each proof of that batch with zeros, deletes it, and lays a "failed" in its place; the batch's
  clients are told that the policy is on hold, and, once the wipe is done and the hold lifted, that
  there is no proof. Every other outcome, such as no verdict, a time limit, a fault in the set-up or
  a response that looks malformed, says nothing of the response: the batch stays back, nothing is
  overwritten, and the check is tried again at a client's later call; if it goes on failing, the
  policy goes on hold. Either way the front end does not stop `cp_prove`, which goes on proving what
  is queued. After a kill of `cp_prove`, or in an `out/` that was copied or restored, a batch can be
  checked in parts, and a rejection takes back nothing that a client has fetched. A proof that has
  lost a mask and still verifies passes this check. What else it does not catch, and when a hold
  comes: [integrations/LIMITS.md](../integrations/LIMITS.md).
- **The spool directory must exist before you first start `cp_prove`.** `python3
  integrations/proof_middleware.py init CONFIG` creates it, which `cp_prove` does not, and writes
  `mark.txt` into it, with the commitment ID ([The mark file](#the-mark-file)). The commands in
  `integrations/` will not write or delete anything in a directory that has contents but no
  `mark.txt`, or that has another commitment's; a mark with 64 zeros, as `cp_prove` writes it, they
  take as any commitment's. No program changes the mark afterwards, so where `cp_prove` marked a
  directory first, the commands of every commitment's front end work on it, those that delete among
  them. A directory without `mark.txt` that holds only `in/`, an empty `out/` or an empty `stop`, as
  the functions of `driver/spool.py` leave it, counts as empty for `cp_prove`, which marks and
  serves it, but the scripts in `integrations/` reject it. So a front end that uses both on one
  directory must run `init` first, or let `cp_prove` start there first. Three kinds of library
  function in `integrations/spool.py` check less (read in the code). `request_stop()` without a
  commitment ID checks only the file's format, so it writes `stop` into any commitment's marked
  directory. The functions that take an open directory check no mark. Nor do those that only read.
- **To delete one delivered response,** call `delete_response_at()` in `integrations/spool.py` with
  an open `out/` directory and the request ID. It deletes the files in the order given under [A
  response](#a-response).
- **Its clean-up departs from this document.** The command `clean` of
  `integrations/proof_middleware.py` and `sweep` of `integrations/spool.py` find old responses by
  listing `out/`, and delete them by age, whether or not a client has fetched them. `clean` also
  deletes old requests that still wait in `in/` or `in/.tmp/`, so they are never served. It does not
  touch requests in `work/`. A file named `hold` in the spool directory stops both, but a `clean`
  that is already running can still delete the one file that it has reached. After any exit code
  other than 0, do not run either of them.
- **It does not see the service's exit code.** It stops only while a file named `hold` is in the
  policy's spool directory. While the file is there, replies under that policy get no proof and
  stored responses are not served.
  - The [start command](../integrations/vllm/REFERENCE.md#start-the-proving-service) creates the
    file once `cp_prove` has exited with a code other than 0. The plain command in [operator's
    duties](operators-duties.md#serve) creates none, and neither does `cp_prove`. The front end's
    own check creates it too (the first bullet).
  - The file pauses one policy and fixes nothing ([its
    limits](../integrations/LIMITS.md#after-a-fault-nothing-stops-by-itself)). Lifting the hold puts
    every stored response back on offer (not while a wipe is unfinished: the hold then comes back),
    faulty ones too, each behind the check of its batch, which a file that has lost a mask can pass;
    a response whose batch has passed that check comes back at once, with no new check. So only a
    person lifts it, and after a rejection not before the wipe is done
    ([integrations/LIMITS.md](../integrations/LIMITS.md)). A batch that the check has overwritten
    does not come back.
  - Stop the vLLM server too, until a person has decided whether the service may start again.

## Options of the service

For the full command, see [operator's duties, "Serve"](operators-duties.md#serve). Except for `-h`,
`--help`, `--setting` and `-clear`, the production build rejects every call (`cp-prove-environment`)
unless `HOME` and `TMPDIR` each hold the absolute path of an existing directory. "Serve" lists the
other requirements.

| Option | Meaning | Default |
|---|---|---|
| `-spool DIR` | the spool directory's absolute path ([what `cp_prove` checks](#what-a-start-does-to-the-spool-directory)). The directory must exist, with mode 0700, and be empty or have a well-formed `mark.txt`. Keep nothing of your own in it. Do not empty a directory to make it fit: name a new one. | |
| `-batch B` | the largest batch: 1 to 64 requests, and at most the number on the parameter file's `slots` line. On an 80 GB card 64 is not to be relied on: on synthetic weights, one call of one batch of 64 (one run, and five more†) and an attempt at a service at 64 (one run, and one more†) each ended at a failed CUDA call with no proof, and the cause isn't shown. Services at 64 on the real, public weights ended well in all three runs: all 960 proofs accepted, with 930 to 1,058 MiB of the card free at its fullest (866 to 1,090 MiB† in three more runs; row T50; [guide/performance.md](../guide/performance.md)). | 1 |
| `-in-step k` | requests proved in one pass over the weights: 1 to 16, and at most the batch size (with `-overlap` or `-overlap-two` and `-batch 1`: 2) | 1 |
| `-end-when-empty` | exit, as for `stop`, when `cp_prove` is about to take a batch and no request is queued | |
| `-most-batches n` | exit, as for `stop`, after n batches | |
| `-memory WORDS` | use another layout of GPU memory. It should not change any byte of a proof, but no test compares the bytes (row T15 in [the table of all rows](limits-all-rows.tsv)). `WORDS` is the preset `lean`, or a comma-separated list of keywords in which `lean` may appear. If a keyword occurs twice, the later one counts. A comment in `prover/prover/arguments.h` lists the keywords. | |
| `-huge-pages` | ask Linux for huge pages for the host copy of the weights. Without it the machine's own setting decides. On a machine that is short of free 2 MiB blocks the request can make the start take far longer. It should not change any byte of a proof, but no test compares the bytes. Nothing that `cp_prove` prints says that it has made the request, or whether Linux granted it. Without the option a full-size `cp_prove` is slow to exit after its last result: wait for the process to end before you start the next program or read the free memory ([how long an exit takes](../prover/docs/ends-in-short.md#how-long-an-exit-takes-at-full-size)). | off |
| `-overlap` | open one batch's weights during the next batch's passes. It needs more GPU memory, an `-in-step` of at least 2, and a `-memory` value that includes `tensor-rooms`, as `lean` does. `cp_prove` rejects it on a real model's files. | off |
| `-overlap-two` | like `-overlap`, which it turns on, for two batches at once. It helps only if a batch fits in one pass, and it delays every response. | off |
| `-public-weights FILE` | serve a commitment that `cp_commit` created with the same option and the same 32-byte file. Never use the option for weights that must stay hidden, and never name a secret file as `FILE` ([Public weights](operators-duties.md#public-weights)). Under this option the weights are not hidden, and whether the requests of one batch stay hidden from one another has not been argued, so serve other people's requests one request per batch, or not at all. No lasting service has run under the option. | |
| `-keep-for-the-audit` | not an option of the service: `cp_prove` rejects it with `-spool`. While the secret key directory's `audit/` holds anything, `cp_prove` does not start on that directory. | |

† Not on the released programs: [which kinds of result these are](limits.md#on-other-programs).

## One batch without a spool directory

`cp_prove -isk DIR -in DIR -out NEW_DIR` proves one batch and exits. These responses too are
unchecked: run `cp_verify` on each proof before you use it. `-in` holds only files named
`<name>.req`, each a request without a nonce. For each request, `cp_prove` writes `<name>.y` and
`<name>.proof`, or `<name>.failed`, into the `-out` directory. There is no tag or `stop` file.
`HOME` and `TMPDIR` must be set as for the service. What follows comes from reading the code (rows
R24, R28 and R36).

- `cp_prove` checks the `-isk` and `-out` paths just as it does when the service starts, so give
  absolute paths. Under `sudo`, the account that ran `sudo` is trusted along that path, and can
  still redirect it through a directory that it owns ([R24](limits-further.md#r24)). The call locks
  the secret key directory as the service does, so it does not start while a service or another
  single batch runs on that directory.
- `-out` must not exist yet, or must hold nothing but `.tmp`, with a well-formed `mark.txt` of the
  kind `out` and otherwise only what `cp_prove` staged in it. `cp_prove` then takes the directory
  over, and first overwrites and removes what an earlier batch staged there. It rejects any other
  existing directory, and touches nothing in it.
- `cp_prove` stages the responses in `.tmp/` below `-out`, beside a `mark.txt` of its own, whose
  second line is `kind out`. It removes both when the batch ends well. After any other exit `.tmp/`
  can stay, with whatever was staged ([`cp_prove -clear`](faults.md#clear)).
- `cp_prove` opens `-in` by the full path, and follows symbolic links in the directories above it.
  So an account that can write to a directory above it can make `cp_prove` read its requests from
  another place. So name an `-in` directory that no other account can write to, with none above it
  that another account can write to ([R24](limits-further.md#r24)).
