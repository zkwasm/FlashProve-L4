# The layout file (layout.txt): where a request's arrays are placed

The layout file (`layout.txt`) is a public text file in the same folder as the parameter file (`policy.txt`). It says into which oracle messages the arrays of a request's proof go, and where each element is. This page says how cp_prove (the GPU prover) and cp_verify (the verifier) use the file, and how to generate one that packs the arrays tightly. It is for anyone who prepares the configuration files of a model.

Paths on this page are relative to prover/. [../docs/glossary.md](../../docs/glossary.md) explains the terms that this page uses.

## What the file holds

The page on the byte format defines the file: [../docs/format/FORMAT.md](../../docs/format/FORMAT.md), 13.4, step 7, "The layout file"; all tests on it are listed there in 13.8, in the rows `plan-layout`. In short, the file has two kinds of line:

- A group's line, `link moment group class rows life`, declares one oracle message of one moment of one segment. The class says what its rows are: `N`, rows over F; `B`, rows over F that are kept for `life` later segments; `W`, rows over E.
- A block's line, `link moment group array R a K e low row place br bj 0`, places a rectangle of one array (2^a rows from R, 2^e columns from K) in a rectangle of 2^br rows and 2^bj positions of its group.

A line whose first byte is `#`, and a line of blanks, tabs and carriage returns only, are skipped. Every row adds bytes to each column that a proof opens, and time to the encoding of the rows, so fewer rows give a smaller proof and a faster prover.

## How the programs use the file

cp_prove looks for it in a file named layout.txt in the same folder as the parameter file, which is isk/config/: cp_commit copies the three configuration files there from its `-config` folder. The verifier (../verifier) reads the file layout.txt of its config folder; its planning call for a request takes the file with the option `-layout`.

**Without the file** no proof is made or judged: the format has no default rule. cp_prove ends with status 1: the digest in isk.txt's line `ranges` covers layout.txt, so the run ends at `refused (ranges-not-guarded)`, before the default build closes its output. A test build under a test seed, which asks for no such line, prints `policy: plan-layout: <path>: there is no layout file`. A call of cp_verify that judges files ends `NO: plan-layout`. The planning call `-plan arrays`, from which the file is made (below), works without it.

**Checks.** Both programs check the file by the same code of the proof library, each on the arrays that it declares by itself (../blocks/frame/req_frame.h; ../blocks/iop/req_rows.h). Among the rules: every group is in a segment and a moment that the request has; every block is in a group of the file and belongs to an array of that moment; an array lies in one group only, has at least one block, and is kept for as many segments as its last reader needs; the blocks are aligned, do not overlap, do not leave the table and cover their array. cp_verify also asks that every array which a segment reads is placed and still kept.

A file that breaks a rule is rejected under the name `plan-layout`: cp_verify ends `NO: plan-layout`, and the line before gives the reason; cp_prove ends with status 1. A file changed after the commitment ends at `refused (ranges-not-guarded)`, before the output closes. A file that fits that line and is still refused (whoever can write the secret key directory can change both) ends after the close, and the default build then shows only `p258 end 1`. A test build under a test seed prints `policy: plan-layout: <path>: <the same reason>`. One sentence comes earlier than all of these, in every build: that layout.txt is there and cannot be read. One bound differs between the two programs: cp_verify reads a file of less than 2^32 bytes, cp_prove one of at most 2^26.

**A public input.** The file's bytes enter the configuration's digest as they are, and the commitment ID hashes that digest, so two encodings of one layout give two commitment IDs. A layout belongs to a parameter file and a file of quantization scales (`KL.bin`): under others its blocks need not cover the arrays.

## How to generate a layout file

tools/pack_layout.py packs the arrays tightly. It needs python3 and runs on the CPU.

The script takes the arrays from the verifier's own list: a planning call `-plan arrays` prints one line `array: ..` for each array, with the oracle message in which it is to lie. Run the block below from prover/, on a folder of your own that holds a copy of your policy.txt and KL.bin and no layout.txt yet: not on an example of this repository, and not on a configuration that you have committed to. The block works in a new temporary folder and moves the finished file into your folder as its last step; a layout.txt that is already there is left as it is.

```sh
C=<your folder with policy.txt and KL.bin>
L=<log2 of a row's message length>
D=$(mktemp -d) &&
{ ../bin/cp_verify -plan arrays "${C:?}" 128 >"${D:?}/arrays.txt"; [ $? -eq 4 ]; } &&
python3 tools/pack_layout.py --arrays "${D:?}/arrays.txt" "${L:?}" 4 >"${D:?}/layout.txt" &&
mv -n "${D:?}/layout.txt" "${C:?}/layout.txt"
```

A planning call ends with status 4, which the block asks for. `L` is 17 at the small examples and 22 at full size (../prover/examples/TARGET-KITS.md, "The three files"). The last number of the script's call, 4, is how many places at the end of every row stay empty. `L` is the fifth word of the line `group link` of your policy.txt, and the script does not compare it with that file. So a wrong `L` that fits still gives a layout file, at status 0, and only the programs' checks above look at it. The last number must be a power of two smaller than 2^L, so here `L` must be at least 3, and `L` can be at most 31. The script refuses anything else, and some lists that aren't a planning call's (the text at its top names the cases), with one line on standard error, nothing on standard output and status 2. Other such lists give a file at status 0, as a wrong `L` does. On standard error the script ends with the line `pack_layout.py: <n> rows over F, <n> rows over E, <n> blocks`. The temporary folder stays: it is yours to delete.

What binds a commitment to its layout file is the configuration's digest, which is part of what the commitment ID hashes. So a commitment that was made with another layout file does not belong to the new layout.txt. Make the commitment again, whole, with the new file in cp_commit's `-config` folder.

The text at the top of the script gives its usage and its input's lines; lay() describes the packing. examples/target_full_f2 contains such a file.
