# Known errors

This page lists known places where this repository's own words are wrong: a page, a comment in a source file or a printed message that says something untrue, or that claims more than the code does or more than has been proved, checked or run (or, in a few entries, less). The section "Your machine and your files" lists a few things that a script or a program here can do to your files or your machine. There are 21 entries; in 18 of them "Where" names source files alone, mostly a comment or a printed message. Three entries (12, 13 and 14) are about tests of this repository that fail through a fault of the test, not of what it tests.

Unless an entry or its section says otherwise, only the wording is at fault: the programs, the Lean statements and their proofs are as the rest of the documentation describes them. No entry is a known way to make a verifier accept a false claim, and none is by itself a known way to learn a model's weights. Some entries under "Weight hiding" are about advice or comments for whoever runs the prover that, followed or trusted as worded, can put weight hiding at risk on that machine (6, 7 and 10); each says what to do instead. Entry 9 concerns synthetic weights alone.

**Where to start.** If you rely on what the release claims, read entry 1. If you run the prover, read 6, 7, 10 and 18. Within each section the weightier entries come first, roughly.

**How to read an entry.** "Where" gives the file and lines of the faulty words (in "Your machine and your files", sometimes of the code itself), "It says" what they say, "In fact" what is true, with the bottom line first, and "What to do" what to rely on instead. A file with lines in parentheses at the end of an entry is where to read on.

**Words used here.** `cp_commit`, `cp_prove` and `cp_verify` are the committing, the proving and the verifying program. A key such as LIM-11 or R2.10 names a row of `docs/limits-all-rows.tsv`. All of these, and "the default setting", are explained in `docs/glossary.md`. "On other programs" is said of a run on programs built from sources that differ from this repository's; how far that speaks for the released programs: `docs/limits.md`, "What was run on these programs, and what was not". Where an entry tells of a run and says neither that nor "on the released programs", take the run as made on other programs, as that page does.

This is not the list of the system's limits. For what is proved and what is not, what is hidden and what is not, and what has been tested, see `guide/what-is-proved.md`, `docs/limits.md` and `lean/THEOREM.md`.

## What is proved, checked or run: soundness, binding, completeness

### 1. "The bound within a third of the distance" and "the proved lemma" are names: what the programs use is argued on paper

**Where:** `verifier/src/cp_verify.cpp`, lines 2916, 3084; `verifier/src/vf_common.h`, line 593; `lean/OnBytes/Shell/Lines.lean`, lines 13-14, 331.

**It says:** `cp_verify`'s output, other printed lines, comments, tests and pages call one source of the programs' error bounds "the bound within a third of the distance". `verify-on-bytes`, a hand translation of `cp_verify` into Lean from which this repository builds no stand-alone program, and comments, tests and pages under `lean/` call it "the proved lemma". The opening comment of `lean/OnBytes/Shell/Lines.lean` says that the closing lines of `verify-on-bytes` are `cp_verify`'s, character for character, with exceptions that do not name the line `commitment:`. `cp_verify` does not print "the proved lemma", so that line differs between the two programs.

**In fact:** At the default setting no round's error bound rests on this lemma; the count is made for each parameter file, and a file that `cp_verify` admits can put rounds on it when a mask code's rate exceeds the main code's; `cp_verify -plan`'s accounting, and the verdict's printed account where it is printed, give the count. The lemma is proved in the paper that this protocol builds on; these pages claim no proof of that lemma itself in this repository. The Lean library under `lean/` also proves a statement of its own of that kind (`Cb.ld_mca_third`: mutual correlated agreement within a third of the distance, for any code that is linear as the protocol reads it (Lean: `LdCode.Laws`, `lean/Cb/Ld/LdRel.lean`: encoder and reading map linear, encoder injective, reading map reading only declared positions)); it has not been compared with the paper's lemma clause by clause, and no theorem ties it to the programs' count. The programs use an adapted form, which rests in part on an argument on paper, in a technical report that is not public and not refereed.

**What to do:** Read both as names: what the programs use is argued on paper. See "Bound within a third of the distance" in `docs/glossary.md` and row LIM-11 of `docs/limits-all-rows.tsv`.

### 2. Some Lean comments claim more than the library proves

**Where:** among them the places that entries L56 to L58, L60 and L87 of `lean/LIMITS.md` name.

**It says:** Some Lean comments say more than the library shows.

**In fact:** A wrong comment makes no theorem false: Lean checks proofs against statements, not comments. Some comments in the Lean files claim more than is proved: entries L56 to L58, L60 and L87 of `lean/LIMITS.md` tell places that are known. How far the other comments agree with the code is not shown (L23, L55).

**What to do:** Never rely on a comment for what a theorem proves. Read the statement itself; for the soundness theorem on `verify` and its limits, read `lean/THEOREM.md`.

### 3. How many theorems behind the soundness proof use the third-party coding theory

**Where:** `lean/Johnson/Goal/Skeleton/SoundOfVerifyAt.lean`, line 42; `lean/Johnson/Goal/Skeleton/KindsLawsAtHead.lean`, lines 31-32; `lean/Johnson/Goal/Skeleton/SoundOfVerifyMore.lean`, lines 42-43; `lean/Johnson/Goal/Skeleton/SoundOfVerifyPolicy.lean`, lines 34-36; `lean/Johnson/Goal/PolicyAndBinding.lean`, line 114.

**It says:** The opening comments of five Lean files, that of the soundness theorem on `verify` among them, say that five theorems used in its proof depend on coding-theory files adapted from third-party Lean (`lean/ThirdParty/`).

**In fact:** No theorem becomes false; the comments undercount. At least six theorems depend on those files. The comments omit `FlashProve.entrySound`: soundness for the copy's file, which commits to a second encoding of the weights and proves that it matches the first. The first comment also lists `Desc.K14.Law.k14_at_free_head` among the theorems that the proof calls; by its own file's opening comment it rests on those files too, which would make seven by that comment's list; by the measure of entry L15 of `lean/LIMITS.md` it lies beneath one of the six.

**What to do:** Use the count in entry L15 of `lean/LIMITS.md`, "Theorems under the statement on codes": it names the six and says what it counts as "called".

### 4. Three Lean comments send the reader to caveats in missing files

**Where:** `lean/Johnson/Flat/Told/Dec/LayoutsAtDescriptionDataAtFiles.lean`, lines 11-12; `lean/Johnson/Flat/Told/Dec/LayoutsAtDescriptionOfFields.lean`, line 10; `lean/Johnson/Flat/Dec/TwoLettersAt.lean`, line 15.

**It says:** The opening comments of `LayoutsAtDescriptionDataAtFiles.lean` and `LayoutsAtDescriptionOfFields.lean` in `lean/Johnson/Flat/Told/Dec/` give their caveats ("every limit") by reference to another file's opening comment (its "head"). That of `lean/Johnson/Flat/Dec/TwoLettersAt.lean` says: "Limits: those of the original's head (not in this library)."

**In fact:** Only comments are affected: Lean checks these files' proofs, not their comments. Each of the three is a changed copy (a "twin", in the comments' word) of a file that is not in this repository, as the comments themselves say. Of their caveats, only the sentences that the first two comments quote are here; the third quotes none. By the import lines, the file of the soundness theorem on `verify` imports all three indirectly.

**What to do:** Read the quoted sentences. Those of `LayoutsAtDescriptionDataAtFiles.lean` send on to the opening comment of `TAtLaws.lean` beside it, which points to entry L57 of `lean/LIMITS.md`, "The limits of two conditional files beneath the soundness theorem on `verify`". `TwoLettersAt.lean` quotes nothing: its caveats are written nowhere here; read its statements, and take nothing from that line.

### 5. Lean comments in three files misdescribe the line `h_par=` of `cp_verify`

**Where:** `lean/OnBytes/Model/Config.lean`, line 836; `lean/OnBytes/Model/TestConfig.lean`, lines 286, 291; `lean/OnBytes/Model/TestRecordWork.lean`, lines 18, 273.

**It says:** Comments in these three files say that the digest made by Lean's `hPar` is the line `h_par=` of `cp_verify`. The one in `Config.lean` adds that `configuration=` prints another digest of the same files; the one in `TestConfig.lean`, that a run of the C++ verifier can confirm the digest.

**In fact:** `cp_verify` prints under `h_par=` the digest that it also prints under `configuration=`, word by word and not byte by byte (`verifier/src/cp_verify.cpp`, `hParOf`; `verifier/docs/parameter-file.md`). Lean's `hPar` makes another digest, with the second hash function. For the same three files `docs/verifying.md` shows `cp_verify`'s line beginning 0adb76ee, and `lean/OnBytes/Model/TestRecordWork.lean` holds a value beginning 17698ba0. `verify-on-bytes` prints its own digest under `h_par=`, so that line too differs from `cp_verify`'s; the opening comment of `lean/OnBytes/Shell/Lines.lean` and `lean/OnBytes/LIMITS.md`, which list where the two differ, do not name it. No verifier compares the printed line `h_par=` with anything, so this bears on nothing that is proved or hidden.

**What to do:** For what `h_par=` is, read `verifier/docs/parameter-file.md`, and take nothing from these comments on it.

## Weight hiding

### 6. Symbolic link in a path: the error message's advice lacks a condition

**Where:** `prover/common/walked_path.h`, line 85; `prover/hash_commitment/hf_commit_making.h`, line 1341; `prover/prover/isk.cuh`, line 77; `tools/audit/audit_batch.sh`, line 75.

**It says:** When a path given to `cp_prove` (`-spool`, `-out`, `-clear`, `-isk`) or `cp_commit` contains a symbolic link, or a folder named to `tools/audit/audit_batch.sh` is one, the error message advises giving the directory itself or its real path instead, with no condition.

**In fact:** The advice is safe only if every link on the way is your own or root's, as `docs/service.md` and `docs/faults.md` say. Another account's link leads wherever that account chooses. With `-isk`, which names the secret key directory, that can be an older copy of it, with its older use count. A use counter set back is one of the two main risks to weight hiding, and no program detects it (`docs/limits.md`, "Where it stands"). The commitment is then "in doubt" (`docs/faults.md`) and must be replaced. All of that is about serving. At `tools/audit/audit_batch.sh` the advice comes before the script has looked at whose link it is, so the folder named on it can be one that another account chose. The script then refuses a path of which a part, or the folder itself, is another account's; a folder of your own to which the link pointed passes. From reading the code.

**What to do:** If a link is another account's, pick a location with no symbolic link in its path: `docs/service.md`, "What `cp_prove` does at startup", and `docs/faults.md`, "`refused (cp-isk)`". (`docs/service.md`, line 153)

### 7. A missing `fresh.bin`: an error message and `docs/faults.md` disagree

**Where:** `prover/prover/isk.cuh`, line 685; `docs/faults.md`, line 117.

**It says:** If `fresh.bin` (32 secret random bytes in the secret key directory) is missing or changed, `cp_prove`'s message `cp-leaf-record` conditionally allows restoring it; `docs/faults.md` says "Do not restore or remove anything."

**In fact:** Not known: which of the two is right. So restore nothing. With the default build you see only the name `cp-leaf-record`: it has closed its output by then, and only a test build prints the long message that allows restoring. That is from reading the code, not from a run.

**What to do:** Restore nothing, as the row for `cp-leaf-record` says in `prover/docs/ends-in-short.md`, "After status 1". (`prover/docs/ends-in-short.md`, line 48)

### 8. Comments in `lean/ZKCP/Defs.lean` mislabel two forms of index privacy

**Where:** `lean/ZKCP/Defs.lean`, lines 304-305, 418, 428-430, 446, 448.

**It says:** Comments in `lean/ZKCP/Defs.lean`, and in files that follow it, call what `HasIndexPrivacy` states the literal definition of index privacy (weight hiding), and say of the stronger form at the end of that file that it is stronger than that definition.

**In fact:** No theorem's statement changes; this matters only when you compare Lean with the definition. From comparing the texts, not by a theorem: `HasIndexPrivacy` is weaker than the definition of index privacy as `lean/TERMS.md`, note 9, states it, and `HasStrongIndexPrivacyWith` is the definition's form for a named simulator, with two differences that `lean/TERMS.md` lists. So the comments claim too much for `HasIndexPrivacy` and `IsZKCPArg`, which includes it.

**What to do:** Follow note 9 in `lean/TERMS.md`, "Notes to table 2"; that page says it takes precedence where comments differ.

### 9. The messages `cp-kept-for-the-audit` and `p258 -keep-for-the-audit` and the batch audit's last line leave out what follows a NO or NOT WHOLE verdict

**Where:** `prover/prover/isk.cuh`, line 201; `prover/prover/closed_front.h`, line 86; `tools/audit/audit.py`, lines 359, 373.

**It says:** With `-keep-for-the-audit`, `cp_prove` stores a batch's 32 random bytes. Of such a batch its error message `cp-kept-for-the-audit` says: "None of that batch's responses may leave unless it has said YES." For any other answer of the audit it points to the table in `tools/audit/README.md`. Its line `p258 -keep-for-the-audit` says: "hand out none of its responses unless the audit has said YES." At NO and NOT WHOLE the batch audit's last line, from `tools/audit/audit.py`, says "None of this batch's responses may be handed out." and nothing of the commitment, while a commitment's own audit at NO prints "Nothing of this commitment may leave."

**In fact:** Only synthetic weights are affected: this option rejects real model files. None of the three says that after a NO or NOT WHOLE verdict nothing of the commitment may ever leave the machine (exception: NOT WHOLE caused by `sudo`). From reading the code, not a run.

**What to do:** After a NO or NOT WHOLE verdict, let nothing of that commitment leave the machine: `docs/operators-duties.md`, "The audit with the key".

### 10. Seven error messages of `cp_commit` and `cp_prove` say "report this line"

**Where:** `prover/hash_commitment/hf_pads.cuh`, line 120; `prover/hash_commitment/hf_commit.cu`, line 250; `prover/hash_commitment/hf_commit_derivation_check.cuh`, lines 150, 155; `prover/prover/batch_keys_held.h`, line 66; `prover/prover/isk.cuh`, line 784; `prover/hiding/trees.cuh`, line 275.

**It says:** Seven error messages of `cp_commit` and `cp_prove` end with "Report this line" or "report this line".

**In fact:** If your weights must stay secret, keep such a line inside the provider, like the rest of what these two programs print: `SECURITY.md` says to send nothing that they printed or logged in a run on such weights. From reading the seven: each is fixed text but for a name, and none prints a count or a value; what such a line can tell is that this check failed. Not known: which of the seven the production build prints.

**What to do:** To report the fault to anyone else, follow SECURITY.md: say in your own words what you saw and where the two programs come from, and send no line, length, exit code or count of that run. If you can, show the problem again on synthetic or public weights and send that run's output. (`docs/operators-duties.md`, "Keep these private")

## File formats, programs, messages and comments

### 11. `ask_and_check.py --resume`: the message on a folder above `--out-dir` advises a new `--out-dir`

**Where:** `integrations/ask_and_check.py`, line 649; `integrations/check_proof.py`, line 353.

**It says:** When a folder above `--out-dir` may be changed by another account, or cannot be examined, `integrations/ask_and_check.py` stops with a sentence that says "Name a new --out-dir below folders that only you or root can change", with `--resume` or without.

**In fact:** That advice is wrong for `--resume`. The client looks above `--out-dir` before it looks at `--resume`, so the advice that knows of `--resume` is not reached. A new `--out-dir` holds no stored run: with `--resume` the client then ends with "does not exist, so there is nothing to resume", and the stored run stays where it was, not resumed. Only to an account that is not root, where the folder is its own and open by its mode alone, does the message first offer `chmod go-w`, which does serve a resume. From reading the code; the move below was tried with the tests' stand-in server, within one file system.

**What to do:** If the message offers `chmod go-w`, that serves a resume; then look at what is in the folder (`integrations/LIMITS.md`, "The look above"). If another account started the run, resume as that account. Otherwise do not change a folder that others use: the stored run names no path, so move the `--out-dir` folder whole below folders that only you or root can change, and name it there with `--resume`. A new `--out-dir` is a new run, without `--resume`.

### 12. `tests/new/run_new.py`: the stage `bytes` always ends with "a stage failed"

**Where:** `tests/new/run_new.py`, lines 88-89, 629.

**It says:** The script's help text says that its stage `bytes` runs three programs from `blocks/frame/tests/` on damaged files: `check_test`, `check_key_test` and `check_entry_test`.

**In fact:** Run from this repository, the stage fails every time. The source of `check_key_test` is not here, so the program cannot be started, and the script ends with "run_new.py: a stage failed", status 1. That failure says nothing about the other programs, which the stage runs all the same. These are `every_byte_test`, which flips every bit of every byte of toy proofs, and `check_test` and `check_entry_test`, which flip one bit of every seventh byte of proof files and of leaf files. So, run from this repository, this stage damages no key file. One key file at toy size with one bit flipped is put to the checker, by `small_key_test`, and other tests that ship put key files with faults at sampled places to it; none does to a key file what `check_entry_test` does to a leaf file at every seventh byte.

**What to do:** Expect two fault lines on `frame/tests/check_key_test`, and read the stage's log, `bytes.log` in the work folder, for the other three programs. `blocks/tests/README.md` says the same of the missing program in its row on `damage.h`. (`blocks/tests/README.md`, line 9)

### 13. `driver/tests/test_measure_rate.py` can fail now and then

**Where:** `driver/tests/test_measure_rate.py`, line 193; `driver/tests/stand_in_programs.py`, line 66.

**It says:** Nothing here is worded wrongly: the fault is in the test. Its helper `record()` reads the file `record` line by line as JSON, while the stand-in programs append their lines to that file from processes of their own.

**In fact:** The test can read the file while a stand-in is still writing a line; it then meets half a line and ends with a `JSONDecodeError`. Seen once in four runs; run again, the test passed. This is a fault of the test, not of `driver/measure_rate.py`, which it tests. The cause is from reading the code and from that one failure. Another ending is told in `docs/limits.md`, "What was run on these programs, and what was not": twice, with the machine under load, on this repository's own `driver/`, a timing test found no sample from its stand-in for `nvidia-smi`; run again, the row passed; the cause is not shown.

**What to do:** If the test ends with a `JSONDecodeError`, or finds no sample from the stand-in for `nvidia-smi`, run it again.

### 14. `integrations.unit` fails where the pytest plugin ddtrace is installed

**Where:** `integrations/tests/test_conftest.py`, line 90.

**It says:** Nothing here is worded wrongly: the fault is in the test. The test `test_below_an_open_place_the_tests_stand_aside` in `integrations/tests/test_conftest.py` asserts that a temporary directory stays empty.

**In fact:** The pytest plugin ddtrace, which this repository does not ask for, leaves an empty file there. The test then fails, and with it the row `integrations.unit` of `SUITE.md`. Seen with ddtrace 4.7.1; with that plugin left out the row passed.

**What to do:** If ddtrace is installed and this test is the row's only failure, run the row where that plugin is not installed.

### 15. Two comments name paths that are not in this repository

**Where:** `prover/prover/chain_run.cuh`, line 174; `prover/prover/prover.h`, line 72.

**It says:** A comment in `prover/prover/chain_run.cuh` names a script, "prover/tests/lost_pad_copies.sh". One in `prover/prover/prover.h` names a folder, "prover/tests/tools".

**In fact:** Only comments are affected. Neither path is in this repository: the tests and tools meant are not published.

**What to do:** Do not look for them here, and take nothing from the two paths.

## Your machine and your files

In this section more than the wording is at stake: each entry concerns something that a script or a program here can do to your files or your machine. For most of them, no other page describes it fully.

### 16. A full disk can leave the audit scripts a truncated input list

**Where:** `tools/audit/pubtool.cpp`, lines 52-53.

**It says:** Nothing here is worded wrongly: the fault is in the code. The helper `pubtool` writes its two lists without checking that the writes succeed. The scripts that call it, `tools/audit/audit_batch.sh` and `tools/audit/audit_commitment.sh`, say so in a comment: "pubtool does not check its writes to the .tsv lists".

**In fact:** The fault makes audits fail, not pass. The helper `pubtool.cpp` ignores write errors on its lists, so a full disk can leave a short list, which audits in that work folder reuse. `audit_batch.sh` ends NOT WHOLE and, with `--erase-kept-files`, erases `batch.fresh`, the audit's kept random bytes: that batch cannot be audited again. That NOT WHOLE also puts the commitment in doubt, so nothing of it may leave the machine (`docs/operators-duties.md`, "The audit with the key"): a full disk can cost the commitment, not only the batch (synthetic weights alone: `-keep-for-the-audit` rejects real model files). `audit_commitment.sh` ends NO VERDICT. From reading the code.

**What to do:** After a full disk, name a new work folder. Run with `--rehearsal` before `--erase-kept-files`, as the opening comment of `tools/audit/audit_batch.sh` advises.

### 17. run_target_example.sh can change the permissions of another directory of yours

**Where:** `prover/examples/run_target_example.sh`, lines 266-267.

**It says:** For what another account can do while `prover/examples/run_target_example.sh` creates its work directory, the script cites rows LIM-51.3 and R2.10 of `docs/limits-all-rows.tsv`. Both describe only a swap for an empty directory.

**In fact:** No file changes, and the script stops (exit code 2). If a directory above the work directory becomes writable to another account during the run, that account can move a non-empty directory of yours to that name as the script creates it; that directory's permissions then become 700. A test did this, with the script paused (on other programs).

**What to do:** Put the work directory inside a directory that only you can write to, and change no permissions above it during the run (`docs/limits-all-rows.tsv`, row R2.10).

### 18. A request file that the prover service cannot read may be deleted

**Where:** `prover/prover/spool.h`, line 589.

**It says:** Nothing here is worded wrongly; the entry stands because the loss is silent. `cp_prove`'s service removes some request files from its spool directory without a response, and a production build prints no name for them. The comment above `take()` in `prover/prover/spool.h` and `docs/service.md` state the rule.

**In fact:** A request can vanish unanswered; only files in the service's spool directory are at risk. A scan spares a file only if its attempt to open it fails with `EACCES`. Any other is moved to `work/` and, if unreadable there, deleted; a production build names it nowhere. From reading the code; the deletion was seen once, in a test harness.

**What to do:** Keep a copy of each request outside the spool directory. `docs/service.md`, "A request file", states the rule correctly. (`docs/service.md`, line 121)

### 19. driver/measure_rate.py may accept a work directory inside the secret key directory

**Where:** `driver/measure_rate.py`, lines 785-786.

**It says:** A comment in `driver/measure_rate.py`, on `keep_out_of_the_given_directories()`, says that the script refuses a work directory that is, or lies in, a directory that the call names. It adds that the secret key directory's path "is compared as text alone".

**In fact:** This matters only if you point `--work` inside the secret key directory (`--isk`). Because the `--isk` path is compared as text only, the first sentence does not always hold. If that path begins with two slashes or goes through a symbolic link, and `--work` uses the plain path, the call is accepted and the script writes its files there. From reading the code, not from a run.

**What to do:** Give `--isk` as an absolute path with one leading slash and no symbolic link, and keep `--work` outside every directory that the call names.

### 20. A race in lean_verify_test.sh when you name its work directory

**Where:** `tools/lean-verify/lean_verify_test.sh`, lines 17, 52.

**It says:** The opening comment of `tools/lean-verify/lean_verify_test.sh`, which tests the script that starts the Lean verifier, says that the test writes only in its work directory if no other account can change the path to it. It does not say what can happen otherwise.

**In fact:** This needs another account that can rename entries in a directory on the path you name; your own directories and /tmp's sticky bit prevent that. It could swap the new directory for a symbolic link before the script checks it; the test then writes files and runs copied scripts where the link leads. From reading the code, not tried.

**What to do:** Run it without an argument, or name a directory below one of your own, as the script's opening comment says. (`tools/lean-verify/lean_verify_test.sh`, line 16)

## Sources and credits

### 21. Lean comments cite a work that is not identified

**Where:** `lean/ZKCP/Defs.lean`, line 8; `lean/ZKCP/Adaptive41.lean`, line 7; `lean/CREDITS.md`, line 35. The same words appear in at least 19 files under `lean/`.

**It says:** Comments in Lean files cite "the work that this protocol builds on", and point to `lean/CREDITS.md`, "Further work".

**In fact:** You cannot look that work up. Under that heading `lean/CREDITS.md` says that it is further work of the cited paper's authors, and gives no title, date or link for it. The definitions that other files cite are written out in Lean in `lean/ZKCP/Defs.lean`.

**What to do:** For a cited definition, read `lean/ZKCP/Defs.lean`.

## What is not on this list

The list is not complete. Smaller faults of wording that were judged unable to mislead a reader are not listed: most stale pointers, loose terms, and places that say less than is true. Among them are several dozen comments and printed messages in the programs' source files; these are listed nowhere in this repository.
