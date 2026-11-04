# Security

Report security problems privately. If the site that hosts this repository offers private
vulnerability reporting, use it. Otherwise, open an issue there saying only that you've found a
problem, with no details that would help an attacker.

If the test suite exits with code 3 ("something false was accepted"), a test reports that a checker
accepted input that it should have rejected. That checker is `cp_verify` or one of the `blocks/`
library's own checkers, but a test in which some other program exited with code 3 is reported the
same way. The lines after the word `LOOK` often say which case it is. Some tests have no such line;
for those, read the test's own output, under `screens/` in the suite's work directory. If it is
`cp_verify`, please report it as a security problem. This is the suite's exit code: when `cp_verify`
itself exits with code 3, a partial check has passed. Leave the suite's work directory and the built
programs untouched, because they include the files that were accepted and the program that accepted
them. Outside the suite too: if `cp_verify` or `lean-verify` accepts input that it should have
rejected, please report it as a security problem.

In any report, include the version, the command you ran, and its output. If you are a provider whose
weights must stay secret, this rule does not hold for a run on those weights. Send nothing that
`cp_commit`, `cp_prove` or the tools in `scales/` printed or logged in such a run, no exit code or
count of such a run, not the file named with `-worst-case-table`, and no response that your own
check did not accept, nor what `cp_verify` printed about it with `-show-stages` or on standard
error: these can depend on the weights, and some can contain unmasked hidden values
([operators-duties.md, step 17](docs/operators-duties.md#from-nothing-to-the-first-proof), and "Keep
these private" there). Say what you saw in your own words, and if you can, show the problem again on
synthetic or public weights and send that run's output.
