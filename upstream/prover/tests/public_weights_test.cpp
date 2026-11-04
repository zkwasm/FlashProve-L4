// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the host code of -public-weights, an option of cp_commit and cp_prove for public weights,
// as a production build (`SERVING_BUILD`) compiles it: the seed derived from the option's file
// (prover/common/public_weights.h), the line "seed public" of isk.txt (prover/common/cp_files.h)
// and the marker files, `<model file>.weights-are-public`
// (prover/hash_commitment/public_weights_markers.h).
// A marker's owner and permissions are tested in both checks: check_the_markers() and
// check_the_digests(), which runs on one small model file without tensors that the test writes.
// Other accounts are simulated: the functions get other user IDs for the caller and for the model
// file's owner. Only as root does the test also give its two markers to another account and
// expect both checks to reject them.
// Not tested: check_the_digests() when a read of a model file fails; what the two
// programs do with the option (see prover/tests/public_weights.py, which needs a GPU); their
// argument parsers.
// Build and run, from the repository's top:
//   prover/tests/unit.sh both tests/public_weights_test.cpp <output directory> <new directory>
// The source's path and a relative <new directory> are resolved from prover/, where the test runs.
// The test creates <new directory> (`folder` in the usage text), keeps its mode at 0700, and
// writes only into files/ inside it. Name it below directories that only you or root can change,
// or in a sticky one such as /tmp: an account that can rename it while the test runs chooses
// where the test writes. The test does not clean up: files/ is left with mode 0755, files/model/
// with mode 0777, and markers given away (as root only) keep owner 12345.
// Last line: "public_weights_test: <n> checks: passes" (exit status 0) or
// "public_weights_test: <k> of <n> checks differ" (1). Status 2: the test stopped early, for
// example after a usage error.
#define SERVING_BUILD 1
#include "common/setting.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>
#include "common/cp_files.h"
#include "common/public_weights.h"
#include "hash_commitment/hf_format.h"
#include "hash_commitment/public_weights_markers.h"

static int checks = 0, differ = 0;
static void check(bool holds, const std::string &what) {
  checks++;
  if (!holds) differ++, printf("DIFFERS: %s\n", what.c_str());
}
static bool write(const std::string &path, const std::string &bytes) {
  FILE *const f = fopen(path.c_str(), "w");
  return f && fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size() && !fclose(f);
}
static std::string hex(const uint8_t *b, size_t n) {
  std::string s;
  char two[3];
  for (size_t i = 0; i < n; i++) snprintf(two, sizeof two, "%02x", b[i]), s += two;
  return s;
}

int main(int argc, char **argv) {
  if (argc != 2 || mkdir(argv[1], 0700)) return fprintf(stderr, "usage: public_weights_test <a NEW folder to write into>\n"), 2;
  // Everything is written into a directory inside the new one. Some checks below let everybody
  // write in `top` and top/model, and change modes and owners there. The new directory keeps
  // mode 0700, so no other account but root can enter it and put its own link at such a name.
  const std::string top = std::string(argv[1]) + "/files";
  if (mkdir(top.c_str(), 0700)) return fprintf(stderr, "public_weights_test: cannot make the folder %s: %s\n", top.c_str(), strerror(errno)), 2;
  umask(022); // whatever the caller's umask is, the files that write() creates have no write permission for the group or for others
  if (!freopen("/dev/null", "w", stderr)) return 2; // the lines that rejections print

  // ---- the seed. `known`: words 0 to 7 of the ChaCha20 block (RFC 8439, section 2.3) for the key
  // 00 01 .. 1f and the last four input words "CPPW", 0, 0, 0. It checks which words go in and
  // which come out; prover/tests/coins_rfc8439_test.cpp tests the block function.
  std::string counting(32, '\0');
  for (int i = 0; i < 32; i++) counting[i] = (char)i;
  const std::string known = "a89484a3cb64f9507febd91a16aa2ee989a2e35d7c98fd3ece6bcab6df2fa03c";
  uint8_t seed[32] = {0};
  public_weights::derive_the_seed(seed, (const uint8_t *)counting.data());
  check(hex(seed, 32) == known, "the seed derived from the bytes 00 .. 1f is the known one");
  check(write(top + "/32", counting) && write(top + "/31", counting.substr(1)) && write(top + "/33", counting + "x") && write(top + "/0", ""), "the test's files are written");
  memset(seed, 0, sizeof seed);
  check(public_weights::seed_of_the_file((top + "/32").c_str(), seed) && hex(seed, 32) == known, "a file of 32 bytes: the same seed");
  for (const char *name : {"/31", "/33", "/0", "/no-such-file", ""}) { // "": the directory itself
    uint8_t untouched[32];
    memset(untouched, 0x5a, sizeof untouched);
    const bool taken = public_weights::seed_of_the_file((top + name).c_str(), untouched);
    check(!taken && std::all_of(untouched, untouched + 32, [](uint8_t b) { return b == 0x5a; }), std::string("is rejected, and the seed's array is left as it was: ") + (*name ? name : "a directory"));
  }

  // ---- isk.txt
  {
    cp::IskText t, back;
    t.version = 1, t.M = 2, t.keep = 3, t.weights = "/w", t.ranges = std::string(64, 'a'), t.public_seed = true;
    const std::string text = cp::isk_text(t), plain = text.substr(0, text.size() - strlen("seed public\n"));
    check(text.size() > 12 && !text.compare(text.size() - 12, 12, "seed public\n"), "isk.txt ends with the line seed public");
    check(cp::cut_isk_text(text, back) && back.public_seed && !back.test_seed && back.ranges == t.ranges, "the text is read back: seed public, not seed test");
    check(cp::cut_isk_text(plain, back) && !back.public_seed && !back.test_seed, "without the line: neither");
    check(cp::cut_isk_text(plain + "seed test\n", back) && back.test_seed && !back.public_seed, "with the line seed test: that one alone");
    for (const char *tail : {"seed public\nseed test\n", "seed test\nseed public\n", "seed public\nseed public\n", "seed public", "seed  public\n", "seed Public\n", "seed\n"})
      check(!cp::cut_isk_text(plain + tail, back), "is no text of isk.txt: it ends otherwise");
  }

  // ---- the markers
  using hf::making::check_the_markers;
  using hf::making::compare_sha256_of_open_file;
  using hf::making::read_public_weights_marker;
  const std::string a = top + "/a.gguf", b = top + "/b.gguf", marker = a + ".weights-are-public";
  const std::string of_abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"; // SHA-256 of "abc" (FIPS 180-4's first example)
  check(write(a, "abc") && write(b, "abc"), "two stand-ins for model files are written");
  const uid_t me = geteuid(), other = me + 1, third = me + 2; // every file of this test belongs to 'me'
  uid_t caller = me, file_owner = me;
  // Reads a.gguf's marker, opened as in check_the_markers_as(), for the accounts `caller` and
  // `file_owner`. says(wanted): the marker is rejected with a message containing `wanted`;
  // says(""): it is accepted and declares `of_abc`.
  auto says = [&](const std::string &wanted) {
    std::string declared;
    const int fd = open(marker.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW);
    const std::string why = read_public_weights_marker(fd, errno, a, caller, file_owner, declared);
    return wanted.empty() ? why.empty() && declared == of_abc : why.find(::gguf::shown(wanted)) != std::string::npos && declared.empty(); // the messages show paths escaped
  };
  check(says("there is no regular file " + marker + ". ") && says("If they are public, create " + marker + ", which must hold what sha256sum prints for " + a + ", with the first command in docs/operators-duties.md, \"Keys, seeds and machines\", \"Declaring a model file's weights public\", and run this command again. ") && !says("emove "), "no marker: the message names it and points to the page's command");
  check(!mkdir(marker.c_str(), 0700) && says(marker + " is not a regular file. ") && says("Remove nothing on this program's word: find out what it is. ") && !says("emove " + marker) && !says("create " + marker), "a directory of that name: the message does not say to remove it");
  check(!rmdir(marker.c_str()) && !symlink("elsewhere", marker.c_str()) && says(marker + " is a symbolic link, which is not accepted. "), "a link that leads nowhere");
  check(write(top + "/elsewhere", of_abc + "  a.gguf\n") && says(marker + " is a symbolic link, which is not accepted. ") && says("Remove nothing on this program's word: find out what it is. ") && !says("emove " + marker) && !says("create " + marker) && check_the_markers({a}) == 2, "a link to a good marker is rejected too, and the message does not say to remove it");
  {
    std::string declared;
    const std::string second = read_public_weights_marker(-1, EMLINK, a, me, me, declared);
    check(second.find(::gguf::shown(marker) + " has a second name (a hard link)") != std::string::npos && second.find("Remove nothing on this program's word") != std::string::npos &&
              second.find("emove " + marker) == std::string::npos && second.find("create " + marker) == std::string::npos && declared.empty(),
          "a marker that was not opened because it has a second name: the message says so, and does not say to remove or to overwrite it");
    check(read_public_weights_marker(-1, EACCES, a, me, me, declared).find(::gguf::shown(marker) + " cannot be read: ") != std::string::npos && declared.empty(), "a marker that cannot be opened: the message gives the cause");
    check(!read_public_weights_marker(-1, 0, a, me, me, declared).empty() && declared.empty(), "no descriptor and no cause: rejected all the same");
  }
  check(!unlink(marker.c_str()) && write(top + "/other", "no digest") && !link((top + "/other").c_str(), marker.c_str()) && says(marker + " has a second name (a hard link), which is not accepted. ") &&
            !says("emove " + marker) && !says("create " + marker) && check_the_markers({a}) == 2,
        "an opened marker with a second name is rejected at the first look, whatever it holds");
  check(!unlink(marker.c_str()) && write(marker, "") && says(" is empty or does not start with 64 lower-case hexadecimal digits") && !says("emove " + marker) && !says("create " + marker), "an empty file: the message does not say to remove or to replace it");
  for (const std::string &text : {of_abc.substr(1) + "  a.gguf\n", of_abc + "0", of_abc + "\t", "BA" + of_abc.substr(2) + "\n", " " + of_abc})
    check(write(marker, text) && says(" is empty or does not start with 64 lower-case hexadecimal digits"), "a malformed marker is rejected");
  for (const std::string &text : {of_abc, of_abc + "\n", of_abc + "  a.gguf\n", of_abc + " *other-name\n"})
    check(write(marker, text) && says(""), "the digest alone, or followed by a space or a newline, is accepted");
  check(check_the_markers({a}) == 0 && check_the_markers({}) == 0, "each listed file has its marker: 0");
  check(check_the_markers({a, b}) == 2, "a second model file without its marker: 2");

  // ---- a marker's owner and permissions. Here a.gguf has a good marker.
  using hf::making::check_the_digests;
  using hf::making::check_the_digests_as;
  using hf::making::check_the_markers_as;
  using hf::making::marker_owner_counts;
  check(marker_owner_counts(5, 5, 6) && marker_owner_counts(6, 5, 6) && marker_owner_counts(0, 5, 6), "a marker of the caller, of the model file's owner or of root counts");
  check(!marker_owner_counts(7, 5, 6) && !marker_owner_counts(7, 0, 6) && !marker_owner_counts(7, 5, 0), "a marker of a fourth account does not count, whoever the caller and the model file's owner are");
  caller = me, file_owner = other;
  check(says(""), "the caller's own marker next to another account's model file is accepted");
  caller = other, file_owner = me;
  check(says(""), "the marker of the model file's owner is accepted, whoever the caller is");
  caller = other, file_owner = third;
  if (me)
    check(says(marker + " belongs to another account (user ID " + std::to_string(me) + "), and counts only if ") && says(" must stay hidden, omit -public-weights, and find out who else writes into that directory. ") &&
              says(" or to root. Nothing was touched. Remove nothing on this program's word. ") && says("If they are public, ask that account to take its file away, and the owner of " + a + ", or root, to create it: their file counts for every account that uses that directory") &&
              !says("emove " + marker) && !says("create " + marker),
          "a marker that belongs neither to the caller nor to the model file's owner nor to root is rejected, and the message says what to do in either case");
  else
    check(says(""), "the test runs as root: its marker is root's and is accepted");
  caller = file_owner = me;
  check(check_the_markers_as(other, {a}) == 0, "the check at the start takes the model file's owner from the path: that account's marker is accepted for another caller");
  struct stat is;
  // The changes of modes and owners below do not rely on argv[1]'s mode 0700 alone. Each is made
  // through a descriptor opened with O_NOFOLLOW, which refuses a link at the path's last name, and
  // only if fstat() on it shows a directory of this account, or a regular file of this account
  // that has no second name.
  auto on_own = [&](const std::string &path, auto change) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW);
    const bool done = fd >= 0 && !fstat(fd, &is) && is.st_uid == me && (S_ISDIR(is.st_mode) || (S_ISREG(is.st_mode) && is.st_nlink == 1)) && !change(fd);
    if (fd >= 0) close(fd);
    return done;
  };
  auto with_mode = [&](const std::string &path, mode_t mode) { return on_own(path, [&](int fd) { return fchmod(fd, mode) || fstat(fd, &is) || (is.st_mode & 0777) != mode; }); };
  auto give_away = [&](const std::string &path) { return on_own(path, [](int fd) { return fchown(fd, 12345, 12345); }); };
  for (mode_t mode : {0664, 0646, 0666, 0620})
    check(with_mode(marker, mode) && says(marker + " can be written by its group or by everybody, ") && says("narrowing the permissions of " + marker + " is not enough: an account that opened the file before can still write to it. See docs/operators-duties.md, \"Keys, seeds and machines\", \"Declaring a model file's weights public\" for what to do instead; if the file is not yours, ask its owner to do it. Then run cp_commit again. ") && says(", the file system keeps no permissions: keep the model files and their .weights-are-public files on one that does") && !says("emove ") && !says("create " + marker) && check_the_markers({a}) == 2,
          "a marker that its group or everybody can write is rejected");
  for (mode_t mode : {0600, 0444, 0644}) check(with_mode(marker, mode) && says("") && check_the_markers({a}) == 0, "a marker that only its owner can write, or nobody, is accepted");
  for (mode_t mode : {0775, 0777, 0755}) check(with_mode(top, mode) && check_the_markers({a}) == 0, "the directory's own permissions are not examined: a shared directory is accepted");
  {
    // A model file that the loader accepts (GGUF, version 3): no tensor, and the three metadata
    // entries that it requires. Otherwise open_folder() would call exit(2).
    auto u32 = [](uint32_t v) { return std::string((const char *)&v, 4); };
    auto u64 = [](uint64_t v) { return std::string((const char *)&v, 8); };
    auto text = [&](const std::string &s) { return u64(s.size()) + s; };
    const std::string bytes = "GGUF" + u32(3) + u64(0) + u64(3) + text("general.architecture") + u32(8) + text("t") + text("t.embedding_length") + u32(4) + u32(1) + text("t.block_count") + u32(4) + u32(1);
    ::Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    const std::string model = top + "/model", its_marker = "/m.gguf.weights-are-public";
    // If mkdir fails, this run has not made top/model, and changes no mode or owner in it.
    const bool made = !mkdir(model.c_str(), 0755) && write(model + "/m.gguf", bytes) && write(model + its_marker, hash.hex() + "  m.gguf\n");
    check(made, "a directory with one small model file and its marker is written");
    ::gguf::Model opened;
    opened.open_folder(model);
    check(check_the_digests(opened, model) == 0, "the loader's check accepts the file and its marker");
    check(check_the_digests_as(other, opened, model) == 0, "for another caller too: the marker belongs to the owner of the opened model file");
    check(made && with_mode(model, 0777) && check_the_digests(opened, model) == 0, "and in a directory that everybody can write");
    check(made && with_mode(model + its_marker, 0664) && check_the_digests(opened, model) == 2, "the loader's check rejects a marker that its group can write");
    check(made && with_mode(model + its_marker, 0644) && check_the_digests(opened, model) == 0, "and accepts it again without that permission");
    // with_mode() itself, on two names that lead to b.gguf, whose mode must stay as it is.
    auto b_is_as_it_was = [&] { return !stat(b.c_str(), &is) && (is.st_mode & 0777) == 0644; };
    check(!symlink("b.gguf", (top + "/by-link").c_str()) && !with_mode(top + "/by-link", 0666) && b_is_as_it_was(), "the test sets no mode through a symbolic link");
    check(!link(b.c_str(), (top + "/second-name").c_str()) && !with_mode(top + "/second-name", 0666) && b_is_as_it_was(), "the test sets no mode on a file that has a second name");
    {
      // A file name in the weights directory can hold any bytes. Both checks must print it escaped
      // (gguf::shown()): one line of printable ASCII. No file gets such a name here: the first
      // check is asked for a file that does not exist, the loader's check gets these bytes at the
      // end of its `directory`, which it uses in messages only.
      const std::string raw = "/\x1b[2J\n\x9b\\", escaped = "/\\x1b[2J\\x0a\\x9b\\x5c";
      auto printed = [](auto call) { // what `call` writes to standard error
        int ends[2];
        const int kept = dup(2);
        if (kept < 0 || pipe2(ends, O_CLOEXEC | O_NONBLOCK)) return std::string(); // O_NONBLOCK: a pipe too small fails the check, and nothing blocks
        fflush(stderr), dup2(ends[1], 2), close(ends[1]);
        call();
        fflush(stderr), dup2(kept, 2), close(kept);
        std::string all(65536, '\0'); // a pipe holds that much by default, and a message is far shorter
        const ssize_t got = read(ends[0], &all[0], all.size());
        close(ends[0]);
        all.resize(got > 0 ? (size_t)got : 0);
        return all;
      };
      auto one_plain_line_with = [](const std::string &line, const std::string &wanted) {
        return line.find(wanted) != std::string::npos && line.back() == '\n' && std::all_of(line.begin(), line.end() - 1, [](unsigned char c) { return c >= 0x20 && c < 0x7f; });
      };
      int first = 0, second = 0;
      check(one_plain_line_with(printed([&] { first = check_the_markers({top + raw + ".gguf"}); }), "there is no regular file " + ::gguf::shown(top) + escaped + ".gguf.weights-are-public. ") && first == 2,
            "control characters, a byte above 0x7e and a backslash in a model file's name: the first check prints them escaped");
      check(write(model + its_marker, std::string(64, '0') + "\n") &&
                one_plain_line_with(printed([&] { second = check_the_digests(opened, model + raw); }), "the SHA-256 of " + ::gguf::shown(model) + escaped + "/m.gguf differs from the one in " + ::gguf::shown(model) + escaped + "/m.gguf.weights-are-public, ") && second == 2,
            "a model file whose SHA-256 is not the marker's is rejected, and the loader's check prints such a path escaped");
      ::Sha256 again; // hex() can be called only once on an object, and `hash` has had its call
      again.update(bytes.data(), bytes.size());
      check(write(model + its_marker, again.hex() + "  m.gguf\n") && check_the_digests(opened, model) == 0, "with the right SHA-256 in the marker again, the loader's check accepts it");
    }
    // Root can give a file away (unless a container maps no such user ID). The model files stay root's; the caller is a third account.
    if (!me && made && give_away(model + its_marker) && give_away(marker))
      check(check_the_digests_as(54321, opened, model) == 2 && check_the_markers_as(54321, {a}) == 2, "as root: a marker that was given to another account is rejected by both checks");
  }
  {
    const int fd = open(a.c_str(), O_RDONLY);
    check(fd >= 0 && lseek(fd, 2, SEEK_SET) == 2 && compare_sha256_of_open_file(fd, of_abc) == 0 && lseek(fd, 0, SEEK_CUR) == 2, "the open file's SHA-256 is the declared one, whatever the file offset, which does not move");
    check(compare_sha256_of_open_file(fd, std::string(64, '0')) == -1, "another digest: they differ");
    close(fd);
    check(compare_sha256_of_open_file(fd, of_abc) == EBADF, "a read that fails: its errno");
  }
  if (differ) return printf("public_weights_test: %d of %d checks differ\n", differ, checks), 1;
  return printf("public_weights_test: %d checks: passes\n", checks), 0;
}
