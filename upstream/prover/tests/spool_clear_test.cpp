// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests cp_prove -clear <folder> (Spool::clear() in prover/prover/spool.h), which removes what an
// earlier cp_prove left in its staging directory: <folder>/out/.tmp (-spool), <folder>/.tmp (-out).
// A start and a bad end, of a service or of a single batch, must clear in the same way. Also tests
// mark.txt, by which -clear and a start know a directory of cp_prove's. CPU only. Build and run:
//   prover/tests/unit.sh both tests/spool_clear_test.cpp <output directory>
// Each case (`Scene`) creates a state in a new directory (`folder` in code and printed lines),
// makes the call, and compares exit status, printed lines and a picture of the directory with what
// it expects. As negative controls, five faulty stand-ins must fail the cases that main() lists; no
// build of cp_prove has them or the test's failing fsync(). The call removes files. So all that the
// test creates, removes or names in a call is in its own new directory under TMPDIR (default /tmp),
// the working directory is two levels inside it (a relative name must be rejected; one that was not
// would still lead there), and the first two calls must leave alone what is not cp_prove's, or the
// test stops. Each call walks its path from /: if cp_prove rejects a directory above the test's, no
// case is run, and nothing has been written in the new directory. If all checks pass, the test
// removes its directory; otherwise it stays and its path is printed (not after the alarm, an
// uncaught exception or a failed chdir("/")). Not tested: cp_prove's main(); another user's
// directories, so nothing under sudo; a network file system; as root, what the system denies (a
// line says so); that a front end writes the same bytes of mark.txt; that cp_prove stages only what
// the call removes. The names of a response's files (y, proof, failed) come from reading spool.h, not
// from a run of cp_prove.
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>
// Replaces fsync() in the header: fails with EIO for the inode number sync_fails_for (0: none).
// Before that, once, it calls at_the_next_sync, if set: what another program does at that moment.
static ino_t sync_fails_for = 0;
static void (*at_the_next_sync)() = nullptr;
static int failing_fsync(int fd) {
  if (void (*now)() = at_the_next_sync) at_the_next_sync = nullptr, now();
  struct stat s;
  if (!fstat(fd, &s) && s.st_ino == sync_fails_for) return errno = EIO, -1;
  return fsync(fd);
}
#define fsync failing_fsync
#include "prover/closed_front.h"
#include "prover/spool.h"
#undef fsync
#include <climits>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

namespace fs = std::filesystem;
static int checks = 0, differ = 0;
static bool loud = true;                // false while a control runs
static std::set<std::string> differing; // the names of the cases that failed in the current run
static void check(bool holds, const std::string &scene, const std::string &what) {
  checks += loud; // a control's run counts no checks: `main` checks which cases it failed
  if (holds) return;
  differing.insert(scene);
  if (loud) differ++, printf("DIFFERS: %s: %s\n", scene.c_str(), what.c_str());
}
static std::string own; // the test's own directory, resolved

// ---- creating a state ----
static void folder(const std::string &path, mode_t mode = 0700) {
  fs::create_directories(fs::path(path).parent_path()), mkdir(path.c_str(), mode);
}
static void file(const std::string &path, const std::string &bytes = "") {
  folder(fs::path(path).parent_path());
  std::ofstream(path, std::ios::binary) << bytes;
}
static void link_to(const std::string &target, const std::string &path) {
  folder(fs::path(path).parent_path());
  if (symlink(target.c_str(), path.c_str()))
    printf("a link cannot be laid: %s\n", path.c_str()), differ++;
}
// A name of the form of a request's id in a spool: 64 lower-case hex digits.
static std::string id(int i) {
  char name[65];
  snprintf(name, sizeof name, "%064x", 0xabc000 + i);
  return name;
}
// ZEROS: what cp_prove writes into a mark's third line. A_FRONT_ENDS_ID: what a front end may write
// there; only its form matters here: 64 lower-case hex digits.
static const std::string ZEROS(64, '0'), A_FRONT_ENDS_ID = "0123456789abcdef" + std::string(48, 'e');
// The bytes of mark.txt, written out here and compared with Spool::mark_text() in main().
static std::string mark_bytes(const std::string &kind, const std::string &commitment = ZEROS) {
  return "cp-folder 1\nkind " + kind + "\ncommitment " + commitment + "\n";
}
// An empty spool directory as a service leaves it; without mark.txt, a directory that only looks
// like one.
static void spool_at(const std::string &at, bool marked = true) {
  for (const char *name : {"", "/in", "/in/.tmp", "/work", "/out", "/out/.tmp"}) folder(at + name);
  if (marked) file(at + "/mark.txt", mark_bytes("spool"));
}
// The staging directory of cp_prove -out in the directory 'at', with its mark.txt.
static void out_at(const std::string &at) { file(at + "/.tmp/mark.txt", mark_bytes("out")); }
// "theirs": files that stand for somebody else's. Several cases have links to them.
static void theirs_at(const std::string &at) {
  file(at + "/theirs/y", "their y"), file(at + "/theirs/proof", "their proof"),
      file(at + "/theirs/key.bin", "KEY"), file(at + "/theirs/entry.bin", "ENTRY");
}
// A closed directory of the user's own that is no spool directory: it has no mark.txt, and files
// under the names that cp_prove stages.
static void mine_at(const std::string &at) {
  for (const char *name : {"y", "proof", "failed"}) file(at + "/out/.tmp/" + id(0) + "/" + name, "mine");
  file(at + "/.tmp/a/proof", "mine"), file(at + "/notes.txt", "mine");
}
// What cp_prove writes into a staging directory: five responses ('names'), in the states in which a
// kill can leave them.
static void staged_at(const std::string &tmp, const std::vector<std::string> &names) {
  file(tmp + "/" + names[0] + "/y", "yyyy"), file(tmp + "/" + names[0] + "/proof", "a whole proof");
  file(tmp + "/" + names[1] + "/failed");
  folder(tmp + "/" + names[2]);
  file(tmp + "/" + names[3] + "/y", "yyyy");
  file(tmp + "/" + names[4] + "/proof", "a proof whose y has its name already");
}

// ---- the picture ----
// All entries under a directory, one line each: the path relative to it, the type, and a link's
// target or a file's bytes. Links are not followed; a directory that cannot be read looks empty.
using Picture = std::set<std::string>;
static Picture picture(const std::string &at) {
  Picture all;
  std::error_code fault;
  for (fs::recursive_directory_iterator
           it(at, fs::directory_options::skip_permission_denied, fault), end;
       !fault && it != end; it.increment(fault)) {
    const fs::file_status is = it->symlink_status();
    std::string line = it->path().lexically_relative(at).string() + "\t";
    if (fs::is_symlink(is))
      line += "link\t" + fs::read_symlink(it->path()).string();
    else if (fs::is_directory(is))
      line += "folder";
    else if (fs::is_regular_file(is)) {
      std::stringstream bytes;
      bytes << std::ifstream(it->path(), std::ios::binary).rdbuf();
      line += "file\t" + bytes.str();
    } else
      line += "other";
    all.insert(line);
  }
  if (fault) all.insert("the picture is not whole: " + fault.message());
  return all;
}
// The picture without these paths and all below them.
static Picture without(const Picture &all, const std::vector<std::string> &gone) {
  Picture rest;
  for (const std::string &line : all) {
    const std::string path = line.substr(0, line.find('\t'));
    bool goes = false;
    for (const std::string &one : gone)
      goes = goes || path == one || !path.compare(0, one.size() + 1, one + "/");
    if (!goes) rest.insert(line);
  }
  return rest;
}
// Whether 'path' is a regular file, not a link, with one name, mode 0600 and exactly these bytes.
static bool is_the_mark(const std::string &path, const std::string &bytes) {
  struct stat is;
  if (lstat(path.c_str(), &is) || !S_ISREG(is.st_mode) || is.st_nlink != 1 || (is.st_mode & 07777) != 0600) return false;
  std::stringstream found;
  found << std::ifstream(path, std::ios::binary).rdbuf();
  return found.str() == bytes;
}
static std::string first_difference(const Picture &wanted, const Picture &found) {
  for (const std::string &line : wanted)
    if (!found.count(line)) return "gone or changed: " + line.substr(0, 200);
  for (const std::string &line : found)
    if (!wanted.count(line)) return "still there or new: " + line.substr(0, 200);
  return "";
}

// A path as seen from the working directory, own + "/a/b". The calls must reject such a path.
static std::string rel(const std::string &absolute) { return absolute.substr(own.size() + 5); }

// ---- the lines ----
static const std::string SAYS = "cp_prove -clear: ";
static const std::string FIX = "Fix the cause and run this again.";
static std::string clean(const std::string &tmp) {
  return SAYS + "nothing that cp_prove staged is left under " + tmp + ".\n";
}
static std::string left(const std::string &path) {
  return SAYS + "LEFT, not touched: " + path + "\n";
}
// top: the named directory, resolved. in_a_spool: the entries are under <top>/out/.tmp.
static std::string not_ours(const std::string &top, bool in_a_spool = true) {
  return SAYS + "cp_prove did not make the entries listed as LEFT, or someone gave them a second name. If " + top +
         " is not the folder that was named with -spool or -out, change nothing in it and name that "
         "folder. If it is, give none of them to anyone: each may be a copy of a staged response. Look at "
         "each; when you know what it is, move it out of this folder and run this again." +
         (in_a_spool ? " cp_prove -spool does not start on this folder while they are there." : "") +
         "\n";
}
static std::string no_folder(const std::string &path) {
  return SAYS + "nothing was touched: " + path +
         " is a link or a file, and cp_prove makes only a folder under that name. If another folder "
         "was named with -spool or -out, name that one. If it was this one, look at it (ls -ld) and give "
         "nothing from there to anyone.\n";
}
// Two messages of Spool::open_by_parts(), which walks the named path. Paths are printed as given.
static std::string named_no_folder(const std::string &part) {
  return SAYS + "nothing was touched: " + part +
         " is a link or a file. Name the folder itself that was named with -spool or -out, by its real "
         "path.\n";
}
static std::string others_can_write(const std::string &above, const std::string &part) {
  return SAYS + "nothing was touched: " + above + ", in which " + part +
         " is looked up, can be written by its group or by others (ls -ld shows its mode), so that "
         "somebody else could put another folder there. " +
         // As in Spool::open_by_parts(), root is never told to chmod. There the caller must also
         // own `above`. Here it does: the test creates each directory that a case opens to others.
         (geteuid() ? "If the folder that you mean does lie there, first take that permission from " + above + " (chmod go-w).\n"
                    : "Use a folder below a directory in which only you and root can write, and leave the mode of " + above + " as it is: others may share it.\n");
}
static std::string open_to_others(const std::string &path) {
  return SAYS + "nothing was touched: " + path +
         " belongs to another user, or is open to its group or to others. cp_prove stages responses "
         "only in folders of its own user's with mode 0700. If another folder was named with -spool or "
         "-out, name that one. If it was this one, give nothing from there to anyone. If it is another "
         "user's, run this as the user who ran cp_prove. If it is yours and only the mode is open, find "
         "out who could write there; then, unless you are root, close it (chmod 700) and run this again; "
         "as root, give nothing from it to anyone and start the next service on a new folder made with "
         "mode 700.\n";
}
static std::string running(const std::string &path) {
  return SAYS + "nothing was touched: a cp_prove is running on " + path +
         ". Run this after it has ended.\n";
}
static std::string none_there(const std::string &top) {
  return SAYS + "nothing was touched: there is no " + top + "/out/.tmp and no " + top +
         "/.tmp, the two places where cp_prove stages responses. If " + top +
         " is the folder that was named with -spool or -out, nothing is left to clear.\n";
}
static std::string no_such(const std::string &named) {
  return SAYS + "nothing was touched: there is no folder " + named +
         ". Name the folder that was named with -spool or -out, by its absolute path.\n";
}
static std::string may_still_lie(const std::string &tmp) {
  return SAYS + "So something that cp_prove staged, and did not release, may still be under " +
         tmp + ": give nothing from there to anyone. " + FIX + "\n";
}
// After the lines left(): a name at the top of a staging directory is not one that cp_prove uses.
static std::string odd_names(const std::string &top) {
  return SAYS + "nothing was touched, and what cp_prove staged there stays: the entries listed as LEFT have names that cp_prove "
         "never uses in a staging folder. If " + top + " is not the folder that was named with -spool or -out, change nothing in "
         "it and name that folder. If it is, give none of them to anyone: each may be a copy of a staged response. Look at each; "
         "when you know what it is, move it out of this folder and run this again.\n";
}
static std::string not_from_the_top(const std::string &named) {
  return SAYS + "nothing was touched: " + named +
         " does not begin with /. Name the folder by its full path from /: a relative name starts at the working "
         "directory, and cp_prove cannot tell whether a link led you there (cd follows links).\n";
}
// The lines about mark.txt. problem: what is wrong with the file, in the words of
// Spool::check_mark(); the next lines give some of them.
// NO_MARK: -clear, whatever the directory holds; a start, if the file that it wrote is gone again.
// NOT_EMPTY: a start, if the top holds a name other than in, out, stop and an unfinished mark.txt.
static const std::string NO_MARK = "it has no mark.txt", NOT_EMPTY = "it is not empty and has no mark.txt", MALFORMED = "its mark.txt is malformed";
static std::string of_another_kind(const std::string &kind) {
  return "its mark.txt is of another kind, " + kind + ": the folder serves another purpose";
}
static std::string is_not(const std::string &path, bool a_spool, const std::string &problem) {
  return path + " is not a " + (a_spool ? "spool directory" : "staging folder of cp_prove -out") + ": " + problem;
}
// -clear. top: the named directory. a_spool: its own mark.txt is at fault, and out/.tmp is not
// cleared; otherwise the one in its .tmp. in_a_marked_spool: that .tmp is at the top of a marked
// spool directory.
static std::string not_marked(const std::string &top, bool a_spool, const std::string &problem, bool in_a_marked_spool = false) {
  return SAYS + "nothing was touched: " + is_not(a_spool ? top : top + "/.tmp", a_spool, problem) +
         ". cp_prove -clear removes nothing unless mark.txt says that the folder is cp_prove's (cp_prove -spool writes that file "
         "into its folder, cp_prove -out into .tmp). " +
         (in_a_marked_spool
              ? top + " is a marked spool directory, at whose top cp_prove makes no .tmp. Give nothing from that .tmp to anyone. Look "
                      "at it, move it out of the folder and run this again: until then " + top + "/out/.tmp is not cleared either."
              : (a_spool ? "" : "If " + top + " is out/ of a spool directory, name the spool directory. ") +
                    "If another folder was named with -spool or -out, name that one. If it was this one, give nothing from " + top +
                    (a_spool ? "/out/.tmp" : "/.tmp") +
                    " to anyone, now or later: responses that were staged and never released may be in it.") +
         "\n";
}
static std::string empty_and_not_marked(const std::string &tmp) {
  return SAYS + "nothing is staged under " + tmp + ", and no mark.txt says that it is cp_prove's: it was left as it is. If another "
         "folder was named with -spool or -out, name that one.\n";
}
// -clear where mark.txt cannot be examined. path: the named directory, or its .tmp. tmp: the
// staging directory that is not cleared.
static std::string not_examined(const std::string &path, const std::string &problem, const std::string &tmp) {
  return SAYS + "nothing was touched: " + path + " cannot be examined: " + problem + ". " + FIX +
         " Until then give nothing from " + tmp + " to anyone.\n";
}
// A start (start() below). holds: with NOT_EMPTY, the names at the top of the directory.
static const std::string UNTOUCHED = ". Nothing that was in the folder was touched. ",
                         ANOTHER = "the service's spool directory, or a new, empty folder (mkdir -m 700)";
static std::string no_start(const std::string &spool, const std::string &problem, const std::string &holds = "") {
  return is_not(spool, true, problem) + UNTOUCHED +
         (problem == NOT_EMPTY || problem == NO_MARK
              ? "cp_prove -spool takes a folder that has a spool directory's mark.txt, or an empty one, into which it writes that file. " +
                    (holds.empty() ? "" : "This one holds " + holds + ". ") + "Remove nothing to make it fit: name "
              : "Change nothing in it: name ") +
         ANOTHER + "\n";
}
// A first start on a directory that holds nothing but in, out and stop, one of which is not as a
// client creates it. cause: which one, and why: one of the next five.
static const std::string STOP_NOT_EMPTY = "stop is not an empty file", OUT_NOT_EMPTY = "out/ is not empty",
                         PARTS_NOT_CLOSED = "in/.tmp is not a folder of yours with mode 0700";
static std::string not_closed(const std::string &name) { // name: in or out
  return name + "/ is not a folder of yours with mode 0700 (if a client has made it: run the service and the client both with sudo, or neither)";
}
static std::string no_request(const std::string &entry) {
  return "in/ holds " + entry + ", which is no request: a request is a regular file named <64 hex digits>.req";
}
static std::string ill_laid(const std::string &spool, const std::string &cause) {
  return is_not(spool, true, NO_MARK + ", and " + cause) + UNTOUCHED +
         "Without mark.txt a folder counts as empty if it holds nothing but in/ with waiting requests, an empty out/ and an empty "
         "file stop, and in/, in/.tmp and out/ are yours and closed to others (mode 0700). Name " + ANOTHER + ". Requests that wait in this one, if any, get no response: submit them again\n";
}
// A start on a path that cannot be opened. why: the system's words.
static std::string not_opened(const std::string &spool, const std::string &why) {
  return spool + ", named with -spool, cannot be opened: " + why + "\n";
}
// The start of a single batch in a directory that exists and is not taken.
static const std::string EXISTS =
    "the folder named with -out exists already, and nothing in it was touched. Name a folder that does not exist "
    "yet, and remove nothing by hand. If a cp_prove failed or was killed there, also run cp_prove -clear on that "
    "folder, so that nothing that it staged stays there: that does not make the folder fit for -out again";
// The start of a single batch in the directory of an earlier one, where an entry stays in .tmp.
static const std::string NOT_ALL_REMOVED =
    "the folder named with -out holds what an earlier batch staged, and not all of it could be removed. Run cp_prove -clear "
    "on that folder: it lists what is left";
// A start at which mark.txt could not be examined, written or synced.
static std::string cannot_be_used(const std::string &spool, const std::string &problem) {
  return spool + " cannot be used: " + problem + UNTOUCHED + "Fix the cause and start again on the same folder\n";
}

// ---- the cases ----
struct Scene {
  std::string name;
  std::function<void(const std::string &at)> lay; // creates the state under 'at', a new directory
  std::string named;                              // the directory that the call is given, from 'at'
  int status;
  // status: the expected exit status. gone: paths from 'at' that the call removes, with all below.
  std::vector<std::string> gone;
  std::function<std::string(const std::string &at)> lines; // the lines that it must print
  // true: 'named' is a spool directory, and Spool::open() is tried on the case too.
  bool at_a_start = false;
  std::vector<std::string> made = {}; // paths from 'at' that the call creates
};
using Call = std::function<int(const std::string &folder, std::string &said)>;
constexpr int T = 8;
constexpr uint32_t V = 100;
static int laid = 0;
static std::string new_place() { // a new directory in the working directory
  const std::string at = own + "/a/b/" + std::to_string(++laid);
  folder(at);
  return at;
}
// A start of a service in the form of a Call: status 0 or 1, and open()'s message as a line. The
// service ends at once.
static int start(const std::string &named, std::string &said) {
  Spool s;
  if (s.open(named, T, V)) return 0;
  said += std::string(s.why) + "\n";
  return 1;
}
// Runs a case with 'call'. With also_a_start, a case with at_a_start runs again for Spool::open(),
// which must succeed if and only if the case's status is 0, and must remove the same entries. If a
// name directly in out/.tmp is not an id (64 lower-case hex digits; .tmp apart), open()
// has another message. T, V: open()'s request length and vocabulary.
static void play(const Scene &scene, const Call &call, bool also_a_start) {
  {
    const std::string at = new_place();
    scene.lay(at);
    const Picture before = picture(at);
    for (const std::string &one : scene.gone)
      check(without(before, {one}) != before, scene.name,
            "the scene does not lay what it expects to go: " + one);
    std::string said;
    const int status = call(at + "/" + scene.named, said);
    check(status == scene.status, scene.name,
          "status " + std::to_string(status) + ", wanted " + std::to_string(scene.status));
    check(said == scene.lines(at), scene.name, "it says:\n" + said + "wanted:\n" + scene.lines(at));
    const Picture after = picture(at);
    for (const std::string &one : scene.made)
      check(without(after, {one}) != after, scene.name, "the call does not make: " + one);
    if (std::count(scene.made.begin(), scene.made.end(), "spool/mark.txt")) // the next comparison omits what is made
      check(is_the_mark(at + "/spool/mark.txt", mark_bytes("spool")), scene.name,
            "the new mark.txt is not a regular file with one name, mode 0600 and the right bytes");
    const std::string other = first_difference(without(before, scene.gone), without(after, scene.made));
    check(other.empty(), scene.name, other);
  }
  if (!scene.at_a_start || !also_a_start) return;
  const std::string at = new_place();
  scene.lay(at);
  const Picture before = picture(at);
  bool no_ids_name = false;
  for (const fs::directory_entry &one : fs::directory_iterator(at + "/" + scene.named + "/out/.tmp")) {
    const std::string name = one.path().filename().string();
    no_ids_name = no_ids_name || (name != ".tmp" && (name.size() != 64 || name.find_first_not_of("0123456789abcdef") != std::string::npos));
  }
  Spool s;
  const bool starts = s.open(at + "/" + scene.named, T, V);
  check(starts == !scene.status, scene.name,
        std::string("a start: open() ") + (starts ? "starts" : "refuses: ") + s.why);
  check(starts || strstr(s.why, no_ids_name ? "Nothing was removed." : "Run cp_prove -clear on the folder named with -spool"), scene.name,
        std::string("a start: the refusal's words: ") + s.why);
  check(starts || strstr(s.why, "cp_prove -clear"), scene.name, std::string("a start: the refusal does not name the call: ") + s.why);
  const std::string other = first_difference(without(before, scene.gone), picture(at));
  check(other.empty(), scene.name, "a start: " + other);
}

static std::vector<Scene> scenes() {
  const std::vector<std::string> ids = {id(0), id(1), id(2), id(3), id(4)};
  const std::vector<std::string> plain = {"a", "req_1-B", std::string(64, 'Z'), "0", "-"};
  auto all_of = [](const std::string &tmp, const std::vector<std::string> &names) {
    std::vector<std::string> gone;
    for (const std::string &name : names) gone.push_back(tmp + "/" + name);
    return gone;
  };
  const std::string TMP = "spool/out/.tmp";
  std::vector<Scene> all;
  all.push_back({"every form, in a spool",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), staged_at(at + "/" + TMP, ids);
                 },
                 "spool", 0, all_of(TMP, ids),
                 [=](const std::string &at) { return clean(at + "/" + TMP); }, true});
  all.push_back({"every form, in the folder of -out",
                 [=](const std::string &at) {
                   out_at(at + "/out7"), staged_at(at + "/out7/.tmp", plain),
                       file(at + "/out7/b.y", "yyyy"), file(at + "/out7/b.proof", "named");
                 },
                 "out7", 0, all_of("out7/.tmp", plain),
                 [](const std::string &at) {
                   return clean(at + "/out7/.tmp");
                 }});
  all.push_back(
      {"in/, work/ and the named responses stay",
       [=](const std::string &at) {
         spool_at(at + "/spool"), staged_at(at + "/" + TMP, ids);
         file(at + "/spool/in/" + id(7) + ".req", "waits"),
             file(at + "/spool/in/.tmp/" + id(8), "half written"),
             file(at + "/spool/work/" + id(0) + ".req", "taken");
         file(at + "/spool/out/" + id(9) + ".y", "yyyy"),
             file(at + "/spool/out/" + id(9) + ".tag", "tag"),
             file(at + "/spool/out/" + id(9) + ".proof", "named"), file(at + "/spool/stop");
       },
       "spool", 0, all_of(TMP, ids), [=](const std::string &at) { return clean(at + "/" + TMP); }});
  all.push_back({"nothing is staged",
                 [](const std::string &at) { spool_at(at + "/spool"); },
                 "spool", 0, {},
                 [=](const std::string &at) { return clean(at + "/" + TMP); }, true});
  all.push_back({"the spool's out/ is named: its .tmp has no mark.txt of its own",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), staged_at(at + "/" + TMP, ids);
                 },
                 "spool/out", 1, {},
                 [=](const std::string &at) { return not_marked(at + "/spool/out", false, NO_MARK); }});
  for (const char *typed : {"by-a-link", "by-a-link/", "by-a-link//"})
    all.push_back({std::string("the folder is named through a link, which is not followed: ") + typed,
                   [=](const std::string &at) {
                     spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p"),
                         link_to("spool", at + "/by-a-link");
                   },
                   typed, 1, {}, [](const std::string &at) { return named_no_folder(at + "/by-a-link"); }});
  all.push_back({"a folder above the named one is a link, which is not followed",
                 [=](const std::string &at) {
                   spool_at(at + "/deep/spool"), file(at + "/deep/" + TMP + "/" + id(0) + "/proof", "p"),
                       link_to("deep", at + "/by-a-link");
                 },
                 "by-a-link/spool", 1, {},
                 [](const std::string &at) { return named_no_folder(at + "/by-a-link"); }});
  all.push_back({"a folder two levels above the named one is a link",
                 [=](const std::string &at) {
                   spool_at(at + "/deep/er/spool"), file(at + "/deep/er/" + TMP + "/" + id(0) + "/proof", "p"),
                       link_to("deep", at + "/by-a-link");
                 },
                 "by-a-link/er/spool", 1, {},
                 [](const std::string &at) { return named_no_folder(at + "/by-a-link"); }});
  // Another account can write in a directory above the named one: it could replace the next part.
  for (const int mode : {0777, 0770, 0720, 0702})
    all.push_back({"a folder above the named one has mode 0" + std::to_string(mode / 64) + std::to_string(mode / 8 % 8) + std::to_string(mode % 8),
                   [=](const std::string &at) {
                     spool_at(at + "/wide/spool"), file(at + "/wide/" + TMP + "/" + id(0) + "/proof", "p");
                     chmod((at + "/wide").c_str(), mode);
                   },
                   "wide/spool", 1, {},
                   [](const std::string &at) { return others_can_write(at + "/wide", "spool"); }});
  all.push_back({"two levels above the named one others can write",
                 [=](const std::string &at) {
                   spool_at(at + "/wide/er/spool"), file(at + "/wide/er/" + TMP + "/" + id(0) + "/proof", "p");
                   chmod((at + "/wide").c_str(), 0777);
                 },
                 "wide/er/spool", 1, {},
                 [](const std::string &at) { return others_can_write(at + "/wide", "er"); }});
  all.push_back({"others may read and enter a folder above the named one",
                 [=](const std::string &at) {
                   spool_at(at + "/plain/spool"), file(at + "/plain/" + TMP + "/" + id(0) + "/proof", "p");
                   chmod((at + "/plain").c_str(), 0755);
                 },
                 "plain/spool", 0, {"plain/" + TMP + "/" + id(0)},
                 [=](const std::string &at) { return clean(at + "/plain/" + TMP); }});
  // The exception, a sticky directory such as /tmp: no other account can rename or remove the part.
  all.push_back({"a folder above the named one has mode 1777, and the next part is the caller's own",
                 [=](const std::string &at) {
                   spool_at(at + "/sticky/spool"), file(at + "/sticky/" + TMP + "/" + id(0) + "/proof", "p");
                   chmod((at + "/sticky").c_str(), 01777);
                 },
                 "sticky/spool", 0, {"sticky/" + TMP + "/" + id(0)},
                 [=](const std::string &at) { return clean(at + "/sticky/" + TMP); }});
  all.push_back({"a link in a folder with mode 1777 gets the link's sentence",
                 [=](const std::string &at) {
                   spool_at(at + "/deep/spool"), file(at + "/deep/" + TMP + "/" + id(0) + "/proof", "p"),
                       folder(at + "/sticky"), link_to("../deep", at + "/sticky/by-a-link");
                   chmod((at + "/sticky").c_str(), 01777);
                 },
                 "sticky/by-a-link/spool", 1, {},
                 [](const std::string &at) { return named_no_folder(at + "/sticky/by-a-link"); }});
  all.push_back({"a mistyped name in a folder with mode 1777",
                 [](const std::string &at) { spool_at(at + "/sticky/spool"), chmod((at + "/sticky").c_str(), 01777); },
                 "sticky/spol", 2, {},
                 [](const std::string &at) { return no_such(at + "/sticky/spol"); }});
  all.push_back({"the path goes up and down again",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p");
                 },
                 "spool/../spool/./", 0, {TMP + "/" + id(0)},
                 [=](const std::string &at) { return clean(at + "/" + TMP); }});
  all.push_back({"the name ends in a slash",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p");
                 },
                 "spool/", 0, {TMP + "/" + id(0)},
                 [=](const std::string &at) { return clean(at + "/" + TMP); }});
  all.push_back({"out/.tmp and .tmp are both there",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p"),
                       out_at(at + "/spool"), file(at + "/spool/.tmp/a/y", "yyyy");
                 },
                 "spool",
                 0,
                 {TMP + "/" + id(0), "spool/.tmp/a"},
                 [=](const std::string &at) {
                   return clean(at + "/" + TMP) + clean(at + "/spool/.tmp");
                 }});
  // What mark.txt is for: by mistake the caller names another closed directory of their own. It has
  // no mark.txt, and entries under the very names that cp_prove stages. Nothing is touched.
  all.push_back({"another closed folder of the user's with cp_prove's names in out/.tmp and .tmp",
                 [](const std::string &at) { mine_at(at + "/mine"); },
                 "mine", 1, {},
                 [](const std::string &at) { return not_marked(at + "/mine", true, NO_MARK); }});
  all.push_back({"another closed folder of the user's with .tmp/a/proof and .tmp/a/y",
                 [](const std::string &at) {
                   file(at + "/mine/.tmp/a/proof", "mine"), file(at + "/mine/.tmp/a/y", "mine"),
                       file(at + "/mine/.tmp/a/notes", "mine"), file(at + "/mine/.tmp/b.txt", "mine"),
                       file(at + "/mine/proof", "mine"), folder(at + "/mine/.tmp/c"),
                       file(at + "/mine/.tmp/d/other", "mine"), file(at + "/mine/.tmp/e/failed", "mine");
                 },
                 "mine", 1, {},
                 [](const std::string &at) { return not_marked(at + "/mine", false, NO_MARK); }});
  all.push_back({"another closed folder of the user's: each name in its .tmp could be a response's",
                 [](const std::string &at) {
                   file(at + "/mine/.tmp/a/proof", "mine"), file(at + "/mine/.tmp/a/y", "mine"),
                       file(at + "/mine/notes.txt", "mine");
                 },
                 "mine", 1, {},
                 [](const std::string &at) { return not_marked(at + "/mine", false, NO_MARK); }});
  // All that a service leaves, but no mark.txt, as after a copy that lost the file.
  all.push_back({"a spool directory without mark.txt, and a response is staged",
                 [=](const std::string &at) {
                   spool_at(at + "/spool", false), staged_at(at + "/" + TMP, ids);
                 },
                 "spool", 1, {},
                 [](const std::string &at) { return not_marked(at + "/spool", true, NO_MARK); }});
  all.push_back({"a spool directory without mark.txt, and nothing is staged",
                 [](const std::string &at) { spool_at(at + "/spool", false); },
                 "spool", 0, {},
                 [=](const std::string &at) { return empty_and_not_marked(at + "/" + TMP); }});
  all.push_back({"a spool directory without mark.txt: any name in out/.tmp counts as staged",
                 [=](const std::string &at) { spool_at(at + "/spool", false), file(at + "/" + TMP + "/mark.txt.making.12.34", "cp-fol"); },
                 "spool", 1, {},
                 [](const std::string &at) { return not_marked(at + "/spool", true, NO_MARK); }});
  all.push_back({"one of two staging folders is not marked: neither is cleared",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p"),
                       file(at + "/spool/.tmp/a/y", "yyyy");
                 },
                 "spool", 1, {},
                 [](const std::string &at) { return not_marked(at + "/spool", false, NO_MARK, true); }});
  // The named directory, its out/ or its out/.tmp is open to group or others: nothing is touched.
  for (const char *wide : {"spool", "spool/out", "spool/out/.tmp"})
    all.push_back({std::string("open to others: ") + wide,
                   [=](const std::string &at) {
                     spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p");
                     chmod((at + "/" + wide).c_str(), 0750);
                   },
                   "spool", 1, {},
                   [=](const std::string &at) { return open_to_others(at + "/" + wide); }});
  all.push_back({"open to others: the folder of -out",
                 [](const std::string &at) {
                   out_at(at + "/out7"), file(at + "/out7/.tmp/a/proof", "p"), chmod((at + "/out7").c_str(), 0755);
                 },
                 "out7", 1, {}, [](const std::string &at) { return open_to_others(at + "/out7"); }});
  all.push_back({"the spool's in/ is named, in/.tmp is empty: out/.tmp is left alone",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p");
                 },
                 "spool/in", 0, {},
                 [](const std::string &at) { return empty_and_not_marked(at + "/spool/in/.tmp"); }});
  all.push_back({"the spool's in/ is named, and in/.tmp contains a file: it is left",
                 [](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/spool/in/.tmp/" + id(8), "half written");
                 },
                 "spool/in", 1, {},
                 [](const std::string &at) { return not_marked(at + "/spool/in", false, NO_MARK); }});
  all.push_back(
      {"a spool directory with a staged proof and an empty .tmp at its top: only the proof goes",
       [=](const std::string &at) {
         spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p"),
             folder(at + "/spool/.tmp");
       },
       "spool", 0, {TMP + "/" + id(0)},
       [=](const std::string &at) { return empty_and_not_marked(at + "/spool/.tmp") + clean(at + "/" + TMP); }});
  all.push_back({"out/.tmp and .tmp are both empty and have no mark.txt",
                 [](const std::string &at) { spool_at(at + "/spool", false), folder(at + "/spool/.tmp"); },
                 "spool", 0, {},
                 [=](const std::string &at) {
                   return empty_and_not_marked(at + "/" + TMP) + empty_and_not_marked(at + "/spool/.tmp");
                 }});
  // The states in which an early kill leaves the directory of -out. An unfinished mark.txt stays.
  all.push_back({"an empty .tmp",
                 [](const std::string &at) { folder(at + "/out7/.tmp"); },
                 "out7", 0, {}, [](const std::string &at) { return empty_and_not_marked(at + "/out7/.tmp"); }});
  all.push_back({".tmp with an unfinished mark.txt alone",
                 [](const std::string &at) { file(at + "/out7/.tmp/mark.txt.making.12.34", "cp-folder 1\nki"); },
                 "out7", 0, {}, [](const std::string &at) { return empty_and_not_marked(at + "/out7/.tmp"); }});
  all.push_back({".tmp with mark.txt alone",
                 [](const std::string &at) { out_at(at + "/out7"); },
                 "out7", 0, {}, [](const std::string &at) { return clean(at + "/out7/.tmp"); }});
  all.push_back({".tmp with mark.txt, an unfinished one and a response: only the response goes",
                 [](const std::string &at) {
                   out_at(at + "/out7"), file(at + "/out7/.tmp/mark.txt.making.12.34", "cp-fol"), file(at + "/out7/.tmp/a/proof", "p");
                 },
                 "out7", 0, {"out7/.tmp/a"}, [](const std::string &at) { return clean(at + "/out7/.tmp"); }});
  // A limit: there the call keeps whatever has an unfinished mark.txt's name, and does not list it.
  all.push_back({".tmp with a folder under an unfinished mark.txt's name",
                 [](const std::string &at) {
                   out_at(at + "/out7"), file(at + "/out7/.tmp/mark.txt.making.x/proof", "mine"), file(at + "/out7/.tmp/a/proof", "p");
                 },
                 "out7", 0, {"out7/.tmp/a"}, [](const std::string &at) { return clean(at + "/out7/.tmp"); }});
  // A name at the top of a staging directory that cp_prove does not give: nothing is touched.
  all.push_back({"foreign names in the staging folder",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/y", "yyyy"),
                       file(at + "/" + TMP + "/" + id(0) + "/proof", "p");
                   file(at + "/" + TMP + "/notes.txt", "mine"),
                       file(at + "/" + TMP + "/.tmp/y", "mine"),
                       file(at + "/" + TMP + "/a b/proof", "mine");
                   file(at + "/" + TMP + "/" + std::string(65, 'a') + "/proof", "mine"),
                       file(at + "/" + TMP + "/.hidden/failed", "mine");
                 },
                 "spool", 1, {},
                 [=](const std::string &at) {
                   const std::string tmp = at + "/" + TMP + "/";
                   return left(tmp + ".hidden") + left(tmp + ".tmp") + left(tmp + "a b") +
                          left(tmp + std::string(65, 'a')) + left(tmp + "notes.txt") +
                          odd_names(at + "/spool");
                 },
                 true});
  all.push_back({"foreign names in the staging folder of -out",
                 [=](const std::string &at) {
                   out_at(at + "/out7"), file(at + "/out7/.tmp/a/proof", "p");
                   file(at + "/out7/.tmp/notes.txt", "mine"), file(at + "/out7/.tmp/a b/proof", "mine"),
                       file(at + "/out7/.tmp/" + std::string(65, 'a') + "/proof", "mine");
                 },
                 "out7", 1, {},
                 [=](const std::string &at) {
                   const std::string tmp = at + "/out7/.tmp/";
                   return left(tmp + "a b") + left(tmp + std::string(65, 'a')) + left(tmp + "notes.txt") +
                          odd_names(at + "/out7");
                 }});
  all.push_back({"a foreign name in one of two staging folders: neither is cleared",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p"),
                       out_at(at + "/spool"), file(at + "/spool/.tmp/a/y", "yyyy"),
                       file(at + "/spool/.tmp/notes.txt", "mine");
                 },
                 "spool", 1, {},
                 [](const std::string &at) { return left(at + "/spool/.tmp/notes.txt") + odd_names(at + "/spool"); }});
  all.push_back({"mark.txt in a spool's staging folder is a foreign name",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p"),
                       file(at + "/" + TMP + "/mark.txt", mark_bytes("out"));
                 },
                 "spool", 1, {},
                 [=](const std::string &at) { return left(at + "/" + TMP + "/mark.txt") + odd_names(at + "/spool"); },
                 true});
  // One level down, what is not cp_prove's stays, and what is staged beside it goes.
  all.push_back({"foreign names in a response's folder",
                 [=](const std::string &at) {
                   spool_at(at + "/spool");
                   for (const char *name :
                        {"y", "proof", "proof.bak", ".proof.swp", "notes", "Y", ".tmp"})
                     file(at + "/" + TMP + "/" + id(0) + "/" + name, name);
                   file(at + "/" + TMP + "/" + id(1) + "/failed");
                 },
                 "spool", 1,
                 {TMP + "/" + id(0) + "/y", TMP + "/" + id(0) + "/proof", TMP + "/" + id(1)},
                 [=](const std::string &at) {
                   const std::string in = at + "/" + TMP + "/" + id(0) + "/";
                   return left(in + ".proof.swp") + left(in + ".tmp") +
                          left(in + "Y") + left(in + "notes") + left(in + "proof.bak") +
                          not_ours(at + "/spool");
                 },
                 true});
  all.push_back({"names in a response's folder that no response holds",
                 [=](const std::string &at) {
                   spool_at(at + "/spool");
                   file(at + "/" + TMP + "/" + id(0) + "/key.bin", "mine"),
                       file(at + "/" + TMP + "/" + id(0) + "/3.bin", "mine");
                 },
                 "spool", 1, {},
                 [=](const std::string &at) {
                   const std::string tmp = at + "/" + TMP + "/";
                   return left(tmp + id(0) + "/3.bin") + left(tmp + id(0) + "/key.bin") +
                          not_ours(at + "/spool");
                 },
                 true});
  // At the top of the staging directory again: names that cp_prove gives only elsewhere.
  all.push_back({"a staged file's name at the top of the staging folder",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p");
                   for (const char *name : {"y", "proof", "failed", "key.bin", "3.bin"})
                     file(at + "/" + TMP + "/" + name, "mine");
                 },
                 "spool", 1, {},
                 [=](const std::string &at) {
                   const std::string tmp = at + "/" + TMP + "/";
                   return left(tmp + "3.bin") + left(tmp + "failed") + left(tmp + "key.bin") +
                          left(tmp + "proof") + left(tmp + "y") + odd_names(at + "/spool");
                 },
                 true});
  all.push_back({"in a spool a response's name is an id: old/proof stays",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/old/proof", "p"),
                       file(at + "/" + TMP + "/" + id(0) + "/proof", "p");
                 },
                 "spool", 1, {},
                 [=](const std::string &at) { return left(at + "/" + TMP + "/old") + odd_names(at + "/spool"); },
                 true});
  all.push_back({"foreign names in the folder of -out: no word on -spool",
                 [=](const std::string &at) {
                   out_at(at + "/out7"), file(at + "/out7/.tmp/a/proof", "p"),
                       file(at + "/out7/.tmp/a/core", "mine");
                 },
                 "out7",
                 1,
                 {"out7/.tmp/a/proof"},
                 [](const std::string &at) {
                   return left(at + "/out7/.tmp/a/core") + not_ours(at + "/out7", false);
                 }});
  all.push_back({"a link under a response's name is left, and not followed",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), theirs_at(at),
                       link_to(at + "/theirs", at + "/" + TMP + "/" + id(0));
                 },
                 "spool", 1, {},
                 [=](const std::string &at) {
                   return left(at + "/" + TMP + "/" + id(0)) + not_ours(at + "/spool");
                 },
                 true});
  all.push_back(
      {"a link under a file's name goes, a file with a second name stays, and what they lead to stays",
       [=](const std::string &at) {
         spool_at(at + "/spool"), theirs_at(at),
             link_to(at + "/theirs/proof", at + "/" + TMP + "/" + id(0) + "/proof");
         if (link((at + "/theirs/y").c_str(), (at + "/" + TMP + "/" + id(0) + "/y").c_str()))
           differ++;
         // A link to a directory is not a directory: it is removed, not followed.
         link_to(at + "/theirs", at + "/" + TMP + "/" + id(1) + "/failed");
       },
       "spool", 1, {TMP + "/" + id(0) + "/proof", TMP + "/" + id(1)},
       [=](const std::string &at) {
         return left(at + "/" + TMP + "/" + id(0) + "/y (it has a second name)") +
                not_ours(at + "/spool");
       },
       true});
  all.push_back({"a folder where a file is due",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof/y", "mine"),
                       file(at + "/" + TMP + "/" + id(0) + "/y", "yyyy"),
                       folder(at + "/" + TMP + "/" + id(1) + "/failed");
                 },
                 "spool", 1, {TMP + "/" + id(0) + "/y"},
                 [=](const std::string &at) {
                   const std::string tmp = at + "/" + TMP + "/";
                   return left(tmp + id(0) + "/proof") + left(tmp + id(1) + "/failed") +
                          not_ours(at + "/spool");
                 },
                 true});
  all.push_back({"a file where a folder is due",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0), "mine");
                 },
                 "spool", 1, {},
                 [=](const std::string &at) {
                   return left(at + "/" + TMP + "/" + id(0)) + not_ours(at + "/spool");
                 },
                 true});
  all.push_back({"a pipe under a file's name goes, and the call does not hang; a pipe under "
                 "another name stays",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), folder(at + "/" + TMP + "/" + id(0)), folder(at + "/" + TMP + "/" + id(1));
                   if (mkfifo((at + "/" + TMP + "/" + id(0) + "/proof").c_str(), 0600) ||
                       mkfifo((at + "/" + TMP + "/" + id(1) + "/pipe.x").c_str(), 0600))
                     differ++;
                 },
                 "spool", 1, {TMP + "/" + id(0)},
                 [=](const std::string &at) {
                   return left(at + "/" + TMP + "/" + id(1) + "/pipe.x") + not_ours(at + "/spool");
                 },
                 true});
  all.push_back(
      {"a name with a line feed writes no line of its own",
       [=](const std::string &at) {
         spool_at(at + "/spool"),
             file(at + "/" + TMP + "/x\ncp_prove -clear: nothing is left\x7f\xc3", "mine");
       },
       "spool", 1, {},
       [=](const std::string &at) {
         return left(at + "/" + TMP + "/x\\x0acp_prove -clear: nothing is left\\x7f\\xc3") +
                odd_names(at + "/spool");
       },
       true});
  all.push_back({"more than twenty are left",
                 [=](const std::string &at) {
                   spool_at(at + "/spool");
                   for (int i = 10; i < 35; i++)
                     file(at + "/" + TMP + "/mine." + std::to_string(i), "mine");
                 },
                 "spool", 1, {},
                 [=](const std::string &at) {
                   std::string lines;
                   for (int i = 10; i < 30; i++)
                     lines += left(at + "/" + TMP + "/mine." + std::to_string(i));
                   return lines + SAYS + "And more is LEFT, not touched.\n" +
                          odd_names(at + "/spool");
                 },
                 true});
  all.push_back({"more than twenty are left in a response's folder",
                 [=](const std::string &at) {
                   spool_at(at + "/spool");
                   for (int i = 10; i < 35; i++)
                     file(at + "/" + TMP + "/" + id(0) + "/mine." + std::to_string(i), "mine");
                 },
                 "spool", 1, {},
                 [=](const std::string &at) {
                   std::string lines;
                   for (int i = 10; i < 30; i++)
                     lines += left(at + "/" + TMP + "/" + id(0) + "/mine." + std::to_string(i));
                   return lines + SAYS + "And more is LEFT, not touched.\n" +
                          not_ours(at + "/spool");
                 },
                 true});
  // Nothing is touched. Status 2: nothing is known to be wrong. 1: a link or a file is named.
  all.push_back({"a folder that was never opened",
                 [](const std::string &at) { folder(at + "/new"), file(at + "/new/mine", "mine"); },
                 "new", 2, {}, [](const std::string &at) { return none_there(at + "/new"); }});
  all.push_back({"out/ without .tmp",
                 [](const std::string &at) { folder(at + "/new/out"); },
                 "new", 2, {}, [](const std::string &at) { return none_there(at + "/new"); }});
  all.push_back({"a mistyped name: nothing is made",
                 [](const std::string &at) { spool_at(at + "/spool"); },
                 "spol", 2, {}, [](const std::string &at) { return no_such(at + "/spol"); }});
  all.push_back({"a link that leads nowhere is named",
                 [](const std::string &at) { link_to("gone", at + "/by-a-link"); },
                 "by-a-link", 1, {},
                 [](const std::string &at) { return named_no_folder(at + "/by-a-link"); }});
  all.push_back({"a file is named",
                 [](const std::string &at) { file(at + "/mine", "mine"); },
                 "mine", 1, {}, [](const std::string &at) { return named_no_folder(at + "/mine"); }});
  all.push_back({"the staging folder itself is named",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p");
                 },
                 TMP, 2, {}, [=](const std::string &at) { return none_there(at + "/" + TMP); }});
  // A link or file where cp_prove makes a directory: nothing is touched on either side of the link.
  all.push_back({"out is a link",
                 [=](const std::string &at) {
                   folder(at + "/spool"), file(at + "/theirs/.tmp/" + id(0) + "/proof", "theirs"),
                       link_to(at + "/theirs", at + "/spool/out");
                 },
                 "spool", 1, {},
                 [](const std::string &at) { return no_folder(at + "/spool/out"); }});
  all.push_back({"out/.tmp is a link",
                 [=](const std::string &at) {
                   folder(at + "/spool/out"), file(at + "/theirs/" + id(0) + "/proof", "theirs"),
                       link_to(at + "/theirs", at + "/spool/out/.tmp");
                 },
                 "spool", 1, {}, [=](const std::string &at) { return no_folder(at + "/" + TMP); }});
  all.push_back({".tmp is a link, beside a staging folder that could be cleared",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p"),
                       file(at + "/theirs/a/proof", "theirs"),
                       link_to(at + "/theirs", at + "/spool/.tmp");
                 },
                 "spool", 1, {},
                 [](const std::string &at) { return no_folder(at + "/spool/.tmp"); }});
  all.push_back({".tmp is a file",
                 [](const std::string &at) { file(at + "/out7/.tmp", "mine"); },
                 "out7", 1, {},
                 [](const std::string &at) { return no_folder(at + "/out7/.tmp"); }});
  // States in which cp_prove -out can leave its directory.
  all.push_back({"-out: released files with names of every form",
                 [=](const std::string &at) {
                   out_at(at + "/out7"), file(at + "/out7/.tmp/a/proof", "p");
                   for (const std::string &name : plain)
                     file(at + "/out7/" + name + ".y", "yyyy"),
                         file(at + "/out7/" + name + ".proof", "named");
                   file(at + "/out7/_.failed");
                 },
                 "out7",
                 0,
                 {"out7/.tmp/a"},
                 [](const std::string &at) {
                   return clean(at + "/out7/.tmp");
                 }});
  all.push_back({"-out: ended well, so .tmp is gone",
                 [](const std::string &at) {
                   file(at + "/out7/a.y", "yyyy"), file(at + "/out7/a.proof", "named");
                 },
                 "out7", 2, {}, [](const std::string &at) { return none_there(at + "/out7"); }});
  // To -clear and to a service's start, a file of the user's own at the top of the directory, or a
  // missing in/ or work/, does not matter.
  all.push_back({"-out: a file of the owner's own next to .tmp",
                 [](const std::string &at) {
                   out_at(at + "/out7"), file(at + "/out7/.tmp/a/proof", "p"),
                       file(at + "/out7/notes.txt", "mine");
                 },
                 "out7",
                 0,
                 {"out7/.tmp/a"},
                 [](const std::string &at) {
                   return clean(at + "/out7/.tmp");
                 }});
  all.push_back({"a spool directory with the operator's own file at its top",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p"),
                       file(at + "/spool/service.log", "mine");
                 },
                 "spool", 0, {TMP + "/" + id(0)},
                 [=](const std::string &at) { return clean(at + "/" + TMP); }, true});
  // A limit: of mark.txt only the kind of file and the bytes are checked.
  all.push_back({"mark.txt has a second name and mode 0666",
                 [=](const std::string &at) {
                   spool_at(at + "/spool"), file(at + "/" + TMP + "/" + id(0) + "/proof", "p");
                   if (link((at + "/spool/mark.txt").c_str(), (at + "/second").c_str()) || chmod((at + "/spool/mark.txt").c_str(), 0666)) differ++;
                 },
                 "spool", 0, {TMP + "/" + id(0)},
                 [=](const std::string &at) { return clean(at + "/" + TMP); }, true});
  all.push_back({"a spool directory that was copied without its empty in/ and work/",
                 [=](const std::string &at) {
                   file(at + "/" + TMP + "/" + id(0) + "/proof", "p"),
                       file(at + "/spool/mark.txt", mark_bytes("spool"));
                 },
                 "spool",
                 0,
                 {TMP + "/" + id(0)},
                 [=](const std::string &at) {
                   return clean(at + "/" + TMP);
                 }});
  return all;
}

// ---- the cases on mark.txt ----
// A mark.txt that no call accepts: at the top of a spool directory, or in .tmp of the -out directory.
struct WrongMark {
  std::string name;
  std::function<void(const std::string &at)> lay; // creates the file
  std::string problem;                            // what each call says of it (Spool::check_mark())
};
// path: the file's path from 'at'.
static std::vector<WrongMark> wrong_marks(const std::string &path, bool a_spool) {
  const std::string kind = a_spool ? "spool" : "out", other = a_spool ? "out" : "spool", good = mark_bytes(kind),
                    malformed = MALFORMED, a_link = "its mark.txt is a link", no_file = "its mark.txt is not a regular file";
  auto bytes = [=](const std::string &them) { return [=](const std::string &at) { file(at + path, them); }; };
  return {{"kind " + other, bytes(mark_bytes(other)), of_another_kind(other)},
          {"kind jobs and no commitment: another program's directory", bytes("cp-folder 1\nkind jobs\n"), of_another_kind("jobs")},
          {"version 2", bytes("cp-folder 2" + good.substr(11)), malformed},
          {"an ID of 63 digits", bytes(mark_bytes(kind, std::string(63, '0'))), malformed},
          {"a capital hex digit", bytes(mark_bytes(kind, std::string(63, '0') + "F")), malformed},
          {"one byte more", bytes(good + "\n"), malformed},
          {"no last line feed", bytes(good.substr(0, good.size() - 1)), malformed},
          {"lines that end in \\r\\n", bytes("cp-folder 1\r\nkind " + kind + "\r\ncommitment " + ZEROS + "\r\n"), malformed},
          {"more bytes than are read", bytes(good + std::string(200, 'x')), malformed},
          {"the first line alone", bytes("cp-folder 1\n"), malformed},
          {"an empty file", bytes(""), malformed},
          {"a link to a fitting file", [=](const std::string &at) { file(at + "/fitting", good), link_to(at + "/fitting", at + path); }, a_link},
          {"a link that leads nowhere", [=](const std::string &at) { link_to(at + "/gone", at + path); }, a_link},
          {"a link to a pipe",
           [=](const std::string &at) {
             if (mkfifo((at + "/pipe").c_str(), 0600)) differ++;
             link_to(at + "/pipe", at + path);
           },
           a_link},
          {"a pipe", [=](const std::string &at) { differ += mkfifo((at + path).c_str(), 0600) ? 1 : 0; }, no_file},
          {"a folder", [=](const std::string &at) { folder(at + path); }, no_file}};
}
// Directories without mark.txt: those on which a service does not start, and those that count as
// unused, on which it does.
static std::vector<Scene> first_starts() {
  auto refused = [](const std::string &holds) { // holds: the names at the top
    return [=](const std::string &at) { return no_start(at + "/spool", NOT_EMPTY, holds); };
  };
  auto faulty = [](const std::string &cause) { // cause: the fault in in, out or stop
    return [=](const std::string &at) { return ill_laid(at + "/spool", cause); };
  };
  const std::string waits = "/spool/in/" + id(7) + ".req", odd_request = no_request(id(7) + ".req");
  std::vector<Scene> all;
  all.push_back({"a start: another closed folder of the user's with cp_prove's names in out/.tmp and .tmp",
                 [](const std::string &at) { mine_at(at + "/spool"); }, "spool", 1, {}, refused(".tmp, notes.txt, out")});
  all.push_back({"a start: a folder with one file of the user's",
                 [](const std::string &at) { file(at + "/spool/notes.txt", "mine"); }, "spool", 1, {}, refused("notes.txt")});
  all.push_back({"a start: a folder with nothing but .tmp/a/proof", // Spool::names_in() skips .tmp unless asked for all
                 [](const std::string &at) { file(at + "/spool/.tmp/a/proof", "mine"); }, "spool", 1, {}, refused(".tmp")});
  all.push_back({"a start: there is no such folder", [](const std::string &) {}, "spool", 1, {},
                 [](const std::string &at) { return not_opened(at + "/spool", "No such file or directory"); }});
  // A client may submit requests before the first start: in/ with .tmp and requests counts as unused.
  all.push_back({"a start: only requests wait",
                 [=](const std::string &at) { folder(at + "/spool/in/.tmp"), file(at + waits, "waits"); },
                 "spool", 0, {}, [](const std::string &) { return ""; }, false,
                 {"spool/mark.txt", "spool/work", "spool/out"}});
  all.push_back({"a start: a request is half written: what in/.tmp holds is not looked at",
                 [=](const std::string &at) { file(at + "/spool/in/.tmp/" + id(8), "half written"), file(at + waits, "waits"); },
                 "spool", 0, {}, [](const std::string &) { return ""; }, false,
                 {"spool/mark.txt", "spool/work", "spool/out"}});
  // From here on, a refusal for in, out or stop names the entry and what is wrong with it.
  all.push_back({"a start: requests wait, and in/ holds another name",
                 [=](const std::string &at) { file(at + waits, "waits"), file(at + "/spool/in/notes.txt", "mine"); },
                 "spool", 1, {}, faulty(no_request("notes.txt"))});
  all.push_back({"a start: requests wait, and the top holds another name",
                 [=](const std::string &at) { file(at + waits, "waits"), file(at + "/spool/notes.txt", "mine"); },
                 "spool", 1, {}, refused("in, notes.txt")});
  all.push_back({"a start: in/ holds another name, and so does the top",
                 [=](const std::string &at) { file(at + "/spool/in/notes.txt", "mine"), file(at + "/spool/notes.txt", "mine"); },
                 "spool", 1, {}, refused("in, notes.txt")});
  all.push_back({"a start: in is a link to a folder in which requests wait",
                 [=](const std::string &at) {
                   folder(at + "/spool"), file(at + "/theirs/" + id(7) + ".req", "waits"), link_to("../theirs", at + "/spool/in");
                 },
                 "spool", 1, {}, faulty(not_closed("in"))});
  all.push_back({"a start: in is a file", [](const std::string &at) { file(at + "/spool/in", "mine"); }, "spool", 1, {}, faulty(not_closed("in"))});
  // Only regular files count as requests that wait, and only a directory as in/.tmp.
  all.push_back({"a start: a folder has a request's name", [=](const std::string &at) { file(at + waits + "/notes.txt", "mine"); },
                 "spool", 1, {}, faulty(odd_request)});
  all.push_back({"a start: a link has a request's name",
                 [=](const std::string &at) { file(at + "/theirs/y", "their y"), link_to("../../theirs/y", at + waits); },
                 "spool", 1, {}, faulty(odd_request)});
  all.push_back({"a start: a pipe has a request's name",
                 [=](const std::string &at) {
                   folder(at + "/spool/in");
                   if (mkfifo((at + waits).c_str(), 0600)) differ++;
                 },
                 "spool", 1, {}, faulty(odd_request)});
  all.push_back({"a start: in/.tmp is a file", [=](const std::string &at) { file(at + waits, "waits"), file(at + "/spool/in/.tmp", "mine"); },
                 "spool", 1, {}, faulty(PARTS_NOT_CLOSED)});
  all.push_back({"a start: in/.tmp is a link",
                 [=](const std::string &at) { file(at + waits, "waits"), folder(at + "/theirs"), link_to("../../theirs", at + "/spool/in/.tmp"); },
                 "spool", 1, {}, faulty(PARTS_NOT_CLOSED)});
  for (const char *wide : {"in", "in/.tmp", "out"}) // each directory must be closed to group and others
    all.push_back({std::string("a start: open to its group: ") + wide,
                   [=](const std::string &at) { folder(at + "/spool/" + wide), chmod((at + "/spool/" + wide).c_str(), 0750); },
                   "spool", 1, {}, faulty(!strcmp(wide, "in/.tmp") ? PARTS_NOT_CLOSED : not_closed(wide))});
  // An empty out/, which a front end may create before it reads responses, and an empty file stop
  // count as unused too. A start removes a stop file that is already there.
  all.push_back({"a start: an empty out alone", [](const std::string &at) { folder(at + "/spool/out"); },
                 "spool", 0, {}, [](const std::string &) { return ""; }, false,
                 {"spool/mark.txt", "spool/in", "spool/work", "spool/out/.tmp"}});
  all.push_back({"a start: a stop file alone", [](const std::string &at) { file(at + "/spool/stop"); },
                 "spool", 0, {"spool/stop"}, [](const std::string &) { return ""; }, false,
                 {"spool/mark.txt", "spool/in", "spool/work", "spool/out"}});
  all.push_back({"a start: requests wait beside an empty out and a stop file",
                 [=](const std::string &at) { file(at + waits, "waits"), folder(at + "/spool/out"), file(at + "/spool/stop"); },
                 "spool", 0, {"spool/stop"}, [](const std::string &) { return ""; }, false,
                 {"spool/mark.txt", "spool/in/.tmp", "spool/work", "spool/out/.tmp"}});
  all.push_back({"a start: out holds a name", [](const std::string &at) { file(at + "/spool/out/notes.txt", "mine"); },
                 "spool", 1, {}, faulty(OUT_NOT_EMPTY)});
  all.push_back({"a start: out is a link to an empty folder",
                 [](const std::string &at) { folder(at + "/theirs"), link_to("../theirs", at + "/spool/out"); }, "spool", 1, {}, faulty(not_closed("out"))});
  all.push_back({"a start: out is a file", [](const std::string &at) { file(at + "/spool/out"); }, "spool", 1, {}, faulty(not_closed("out"))});
  all.push_back({"a start: stop is not empty", [](const std::string &at) { file(at + "/spool/stop", "mine"); }, "spool", 1, {}, faulty(STOP_NOT_EMPTY)});
  all.push_back({"a start: stop is a link to an empty file",
                 [](const std::string &at) { file(at + "/empty"), link_to("../empty", at + "/spool/stop"); }, "spool", 1, {}, faulty(STOP_NOT_EMPTY)});
  all.push_back({"a start: stop is a folder", [](const std::string &at) { folder(at + "/spool/stop"); }, "spool", 1, {}, faulty(STOP_NOT_EMPTY)});
  // Of several faults the first by name, in byte order, is reported.
  all.push_back({"a start: in, out and stop are at fault",
                 [](const std::string &at) {
                   file(at + "/spool/in/notes.txt", "mine"), file(at + "/spool/out/notes.txt", "mine"), file(at + "/spool/stop", "mine");
                 },
                 "spool", 1, {}, faulty(no_request("notes.txt"))});
  all.push_back({"a start: out and stop are at fault",
                 [=](const std::string &at) { file(at + waits, "waits"), file(at + "/spool/out/notes.txt", "mine"), file(at + "/spool/stop", "mine"); },
                 "spool", 1, {}, faulty(OUT_NOT_EMPTY)});
  all.push_back({"a start: in/.tmp is a file, and in/ holds another name",
                 [](const std::string &at) { file(at + "/spool/in/.tmp", "mine"), file(at + "/spool/in/notes.txt", "mine"); },
                 "spool", 1, {}, faulty(PARTS_NOT_CLOSED)});
  all.push_back({"a start: in/ holds two other names",
                 [=](const std::string &at) { file(at + waits, "waits"), file(at + "/spool/in/b.txt", "mine"), file(at + "/spool/in/a.txt", "mine"); },
                 "spool", 1, {}, faulty(no_request("a.txt"))});
  return all;
}
// A wrong mark.txt alone in the directory: there the picture would show whatever a start made before
// it asked for the file.
static std::vector<Scene> marks_alone() {
  std::vector<Scene> all;
  for (const WrongMark &wrong : wrong_marks("/spool/mark.txt", true))
    all.push_back({"a start: nothing but a wrong mark.txt: " + wrong.name,
                   [=](const std::string &at) { folder(at + "/spool"), wrong.lay(at); },
                   "spool", 1, {}, [=](const std::string &at) { return no_start(at + "/spool", wrong.problem); }});
  return all;
}

// ---- the controls' stand-ins ----
static bool inside_the_tests_folder(const std::string &path) {
  return !own.empty() && !path.compare(0, own.size() + 1, own + "/");
}
// Does and prints what the true call does, then removes whatever is still in the staging
// directories: only the picture can differ. Each stand-in acts only inside the test's directory.
static int removes_too_much(const std::string &named, std::string &said) {
  if (!inside_the_tests_folder(named)) return 99;
  const int status = Spool::clear(named, said);
  for (const char *tmp : {"/out/.tmp", "/.tmp"}) {
    std::error_code fault;
    if (!fs::is_directory(fs::symlink_status(named + tmp, fault))) continue;
    for (const fs::directory_entry &one : fs::directory_iterator(named + tmp))
      fs::remove_all(one.path(), fault); // it follows no link
  }
  return status;
}
// Does what the true call does, but first moves every staged proof aside and afterwards back, and
// prints that nothing is left.
static int leaves_the_proofs(const std::string &named, std::string &said) {
  if (!inside_the_tests_folder(named)) return 99;
  std::vector<std::string> proofs;
  for (const char *tmp : {"/out/.tmp", "/.tmp"}) {
    std::error_code fault;
    if (!fs::is_directory(fs::symlink_status(named + tmp, fault))) continue;
    for (const fs::directory_entry &one : fs::directory_iterator(named + tmp))
      if (fs::is_regular_file(fs::symlink_status(one.path() / "proof", fault)))
        proofs.push_back(one.path().string());
  }
  // Aside: out of the staging directory, where the true call would reject the new name.
  folder(own + "/aside");
  for (size_t i = 0; i < proofs.size(); i++) rename((proofs[i] + "/proof").c_str(), (own + "/aside/" + std::to_string(i)).c_str());
  std::string unused;
  Spool::clear(named, unused);
  for (size_t i = 0; i < proofs.size(); i++)
    mkdir(proofs[i].c_str(), 0700), rename((own + "/aside/" + std::to_string(i)).c_str(), (proofs[i] + "/proof").c_str());
  for (const char *tmp : {"/out/.tmp", "/.tmp"})
    if (fs::is_directory(named + tmp)) said += clean(named + tmp);
  return 0;
}
// Does what 'call' (-clear or a start) would do if it did not ask for mark.txt: where the file is
// missing from the named directory or its .tmp, it writes one, makes the true call and removes it.
static Call asking_for_no_mark(const Call &call) {
  return [=](const std::string &named, std::string &said) {
    if (!inside_the_tests_folder(named)) return 99;
    std::vector<std::string> written;
    std::error_code fault;
    for (const char *in : {"", "/.tmp"}) {
      const std::string path = named + in + "/mark.txt";
      if (!fs::is_directory(fs::symlink_status(named + in, fault)) || fs::exists(fs::symlink_status(path, fault))) continue;
      file(path, mark_bytes(*in ? "out" : "spool")), written.push_back(path);
    }
    const int status = call(named, said);
    for (const std::string &path : written) unlink(path.c_str());
    return status;
  };
}
// Does what a start would do if it made its subdirectories before it asked for mark.txt.
static int makes_first(const std::string &named, std::string &said) {
  if (!inside_the_tests_folder(named)) return 99;
  for (const char *name : {"/in", "/work", "/out", "/out/.tmp", "/in/.tmp"}) folder(named + name);
  return start(named, said);
}

int main() {
  // As cp_prove does at its start (prover/cp_prove.cu): under sudo, a directory above the work folder may belong to the account that typed sudo, as where the folder lies below that account's home.
  Spool::typed_sudo = take_who_typed_sudo();
  const char *tmpdir = getenv("TMPDIR");
  std::string pattern =
      std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") + "/spool_clear_test.XXXXXX";
  if (!mkdtemp(pattern.data())) return printf("cannot make a work folder\n"), 2;
  char whole[PATH_MAX];
  // The path without links. Its last part must be the name that mkdtemp() has just made: that
  // directory is what is removed at the end.
  if (!realpath(pattern.c_str(), whole) ||
      strcmp(strrchr(whole, '/'), strrchr(pattern.c_str(), '/')))
    return printf("the work folder's path cannot be resolved; the folder stays: %s\n",
                  pattern.c_str()),
           2;
  own = whole;
  // The call shows a byte that is not printable ASCII as \x.., and the expected lines below do not.
  for (const unsigned char c : own)
    if (c < 0x20 || c >= 0x7f)
      return rmdir(own.c_str()),
             printf("not run: the work folder's path holds a byte that is not printable ASCII; set "
                    "TMPDIR to another folder\n"),
             2;
  alarm(600); // a call that hangs, as on a pipe, ends the test; the directory then stays
  umask(077);
  // Before anything is written in the new directory, one call, which creates nothing, walks the
  // path from /. If the walk stops above the directory, another account could redirect the test's
  // paths, so no case runs. The line quotes cp_prove's: unit.sh shows only the log's last line.
  if (std::string said; Spool::clear(own, said) == 1 && said.find(own) == std::string::npos &&
                        (said.find("can be written by its group or by others") != std::string::npos ||
                         said.find("belongs to an account other than root and you") != std::string::npos))
    return printf("not run: cp_prove rejects a folder above this test's work folder: %s Set TMPDIR to a folder in "
                  "which, as in every folder above it, only you or root can write (/tmp with its sticky bit will do "
                  "too); under run_suite.py, which sets TMPDIR itself, name a work folder in such a place. This "
                  "test's work folder stays: %s\n",
                  said.substr(SAYS.size(), said.size() - SAYS.size() - 1).c_str(), own.c_str()),
           2;
  folder(own + "/a/b");
  if (chdir((own + "/a/b").c_str()))
    return printf("the work folder cannot be entered; it stays: %s\n", own.c_str()), 2;
  const Call the_call = [](const std::string &named, std::string &said) {
    return Spool::clear(named, said);
  };

  { // Before all cases: a directory without a staging directory, then a staging directory with one
    // file that is not cp_prove's. If either is not left alone, the test stops.
    const std::string at = new_place();
    file(at + "/new/mine", "mine"), spool_at(at + "/spool"),
        file(at + "/spool/out/.tmp/notes.txt", "mine");
    const Picture before = picture(at);
    std::string said;
    if (Spool::clear(at + "/new", said) != 2 || picture(at) != before ||
        Spool::clear(at + "/spool", said) != 1 || picture(at) != before)
      return printf("The first two calls failed: a folder without a staging folder, or one foreign file in "
                    "out/.tmp, was not left alone, or the status is not 2 and then 1. No further "
                    "scene is run. The folder stays: %s\n",
                    own.c_str()),
             1;
  }
  const std::vector<Scene> all = scenes();
  for (const Scene &scene : all) play(scene, the_call, true);

  { // The call made twice, for -spool and for -out: the staging directory stays, so the second call
    // finds it and prints the same.
    const std::string at = new_place(), scene = "the call made twice";
    spool_at(at + "/spool"), staged_at(at + "/spool/out/.tmp", {id(0), id(1), id(2), id(3), id(4)});
    out_at(at + "/out7"), staged_at(at + "/out7/.tmp", {"a", "b", "c", "d", "e"});
    for (const char *named : {"/spool", "/out7"}) {
      const bool a_spool = !strcmp(named, "/spool");
      const std::string tmp = at + named + (a_spool ? "/out/.tmp" : "/.tmp");
      const Picture cleared = a_spool ? Picture{} : Picture{"mark.txt\tfile\t" + mark_bytes("out")}; // -out: mark.txt stays
      std::string first, second;
      check(!Spool::clear(at + named, first) && first == clean(tmp) && picture(tmp) == cleared &&
                !Spool::clear(at + named, second) && second == first && picture(tmp) == cleared,
            scene, at + named);
    }
    // After what was LEFT is moved out, the next call ends with status 0.
    file(at + "/spool/out/.tmp/notes", "mine");
    std::string said;
    check(Spool::clear(at + "/spool", said) == 1 &&
              !rename((at + "/spool/out/.tmp/notes").c_str(), (at + "/notes").c_str()) &&
              !Spool::clear(at + "/spool", said),
          scene, "what was LEFT is moved out, and the call is made again");
  }
  { // A .tmp that is not cp_prove's at the top of a marked spool directory: the call says to move it
    // out, and then clears.
    const std::string at = new_place(), scene = "a .tmp at the top of a marked spool directory is moved out", spool = at + "/spool";
    spool_at(spool), file(spool + "/out/.tmp/" + id(0) + "/proof", "p"), file(spool + "/.tmp/a/y", "mine");
    const Picture before = picture(at);
    std::string said;
    const int first = Spool::clear(spool, said);
    const bool untouched = picture(at) == before, moved = !rename((spool + "/.tmp").c_str(), (at + "/moved").c_str());
    const int second = Spool::clear(spool, said);
    check(first == 1 && untouched && moved && !second && picture(spool + "/out/.tmp").empty() &&
              picture(at + "/moved") == Picture{"a\tfolder", "a/y\tfile\tmine"},
          scene, said);
  }
  { // A relative name is rejected by -clear and by a start, wherever it leads: here to a spool
    // directory that -clear would clear.
    const std::string at = new_place(), scene = "a relative name";
    spool_at(at + "/spool"), file(at + "/spool/out/.tmp/" + id(0) + "/y", "yyyy");
    const Picture before = picture(at);
    const std::vector<std::pair<std::string, std::string>> names = { // the working directory, the name
        {own + "/a/b", rel(at) + "/spool"}, {own + "/a/b", "./" + rel(at) + "/spool"},
        {own + "/a/b", "../b/" + rel(at) + "/spool"}, {own + "/a/b", "spool"}, // the last leads nowhere
        {at, "spool"}, {at + "/spool", "."}, {at + "/spool/out", ".."}};
    for (const auto &[from, named] : names) {
      std::string clear_says, start_says;
      const bool entered = !chdir(from.c_str());
      const int cleared = Spool::clear(named, clear_says), started = start(named, start_says);
      check(entered && !chdir((own + "/a/b").c_str()) && cleared == 1 && started == 1 && clear_says == not_from_the_top(named) &&
                start_says.find(named + " does not begin with /.") != std::string::npos && picture(at) == before,
            scene, named + ", from " + from + ":\n" + clear_says + start_says);
    }
  }
  { // cp_prove's `main` passes "" for a wrong number of arguments.
    std::string said;
    check(Spool::clear("", said) == 1 &&
              said ==
                  SAYS + "usage: cp_prove -clear <folder>, with no other argument. <folder> is the folder "
                         "that was named with -spool or with -out. Nothing was touched.\n",
          "no folder is named", said);
  }
  { // The locks. A lock taken through a second open of the same directory is denied within one
    // process as it is between two.
    const std::string at = new_place(), scene = "the locks";
    folder(at + "/spool"), folder(at + "/in");
    const char word[4] = {1, 0, 0, 0};
    {
      Spool serves;
      check(serves.open(at + "/spool", T, V) &&
                serves.stage_answer(id(0), {{"y", word, 4}, {"proof", "proof", 5}}),
            scene, "a service opens and stages a response");
      check(is_the_mark(at + "/spool/mark.txt", mark_bytes("spool")), scene, "the service has marked the empty folder");
      const Picture before = picture(at);
      std::string said;
      check(Spool::clear(at + "/spool", said) == 2 && said == running(at + "/spool") &&
                picture(at) == before,
            scene, "a living service, the spool is named: " + said);
      said.clear();
      check(Spool::clear(at + "/spool/out", said) == 2 && said == running(at + "/spool/out") &&
                picture(at) == before,
            scene, "a living service, its out/ is named: " + said);
      serves.drop_what_is_staged();
    }
    {
      Spool one_batch;
      check(one_batch.open_one_batch(at + "/in", at + "/out7", T, V) &&
                one_batch.stage_answer("a", {{"y", word, 4}}),
            scene, "one batch opens and stages a response");
      check(is_the_mark(at + "/out7/.tmp/mark.txt", mark_bytes("out")), scene, "cp_prove -out has marked its new .tmp");
      const Picture before = picture(at);
      std::string said;
      check(Spool::clear(at + "/out7", said) == 2 && said == running(at + "/out7") &&
                picture(at) == before,
            scene, "a living cp_prove -out: " + said);
    }
    std::string said;
    check(!Spool::clear(at + "/out7", said) && said == clean(at + "/out7/.tmp") &&
              picture(at + "/out7/.tmp") == Picture{"mark.txt\tfile\t" + mark_bytes("out")},
          scene, "after a cp_prove -out that ended without tidy() the call clears, and mark.txt stays: " + said);
    // The other way round: while either lock is held, as by a running call, a service does not
    // start and removes nothing.
    file(at + "/spool/out/.tmp/" + id(1) + "/proof", "p");
    for (const char *held : {"/spool", "/spool/out"}) {
      const int fd = open((at + held).c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
      const Picture before = picture(at);
      {
        Spool late;
        check(fd >= 0 && !flock(fd, LOCK_EX | LOCK_NB) && !late.open(at + "/spool", T, V) &&
                  !strcmp(late.why, "another cp_prove serves this spool or clears it") &&
                  picture(at) == before,
              scene, std::string("the lock is held on ") + held + ": " + late.why);
      }
      said.clear();
      check(Spool::clear(at + "/spool", said) == 2 && said == running(at + held) &&
                picture(at) == before,
            scene, "and a second call is turned away: " + said);
      close(fd);
    }
  }
  { // A single batch that ends badly drops what it staged, and whatever else in .tmp has a staged
    // response's names. If only mark.txt is then left in .tmp, that file goes too, and -clear finds
    // nothing to do. If an entry stays, the call fails, mark.txt stays, and -clear lists the entry.
    const std::string at = new_place(), scene = "a single batch drops what it staged";
    folder(at + "/in");
    const char word[4] = {1, 0, 0, 0};
    int n = 7;
    for (const char *beside : {"", "b/proof", "b/notes"}) { // b/proof: a response that the Spool has lost track of
      const bool is_left = !strcmp(beside, "b/notes");
      const std::string out = at + "/out" + std::to_string(n++), tmp = out + "/.tmp";
      const Picture the_mark = {"mark.txt\tfile\t" + mark_bytes("out")};
      bool opened, dropped;
      {
        Spool one_batch;
        opened = one_batch.open_one_batch(at + "/in", out, T, V) &&
                 one_batch.stage_answer("a", {{"y", word, 4}});
        if (*beside) file(tmp + "/" + beside, "mine");
        dropped = opened && one_batch.drop_what_is_staged();
      }
      const Picture after = picture(tmp);
      std::string said;
      const int status = Spool::clear(out, said);
      check(opened && dropped == !is_left && picture(tmp) == after &&
                (is_left ? without(after, {"b"}) == the_mark && after.count("b/notes\tfile\tmine") && status == 1 &&
                               said == left(tmp + "/b/notes") + not_ours(out, false)
                         : after.empty() && !status && said == empty_and_not_marked(tmp)),
            scene,
            std::string("beside the response, ") + (*beside ? beside : "nothing") + ": opened " +
                std::to_string(opened) + ", dropped " + std::to_string(dropped) + ": " + said);
    }
    // A service's bad end clears out/.tmp in the same way.
    for (const char *beside : {"/proof", "/notes"}) {
      const bool is_left = !strcmp(beside, "/notes");
      const std::string spool = at + "/spool" + std::to_string(n++), tmp = spool + "/out/.tmp";
      folder(spool);
      bool opened, dropped;
      std::string why;
      {
        Spool serves;
        opened = serves.open(spool, T, V) && serves.stage_answer(id(0), {{"y", word, 4}});
        file(tmp + "/" + id(1) + beside, "mine");
        dropped = opened && serves.drop_what_is_staged();
        why = serves.why;
      }
      check(opened && dropped == !is_left &&
                picture(tmp) == (is_left ? Picture{id(1) + "\tfolder", id(1) + "/notes\tfile\tmine"} : Picture{}) &&
                is_the_mark(spool + "/mark.txt", mark_bytes("spool")),
            "a service drops what it staged",
            std::string("beside the response, <id>") + beside + ": opened " + std::to_string(opened) + ", dropped " +
                std::to_string(dropped) + ": " + why);
    }
  }
  { // A single batch: what is not taken.
    // Without mark.txt in .tmp a directory that exists is not taken, and nothing is written in it:
    // an empty .tmp, also that of out/ in a spool directory that no service holds.
    for (const bool of_a_spool : {false, true}) {
      const std::string here = new_place(), named = here + (of_a_spool ? "/spool/out" : "/out7");
      folder(here + "/in");
      if (of_a_spool)
        spool_at(here + "/spool");
      else
        folder(named + "/.tmp");
      const Picture as_it_was = picture(here);
      Spool late;
      const bool opened = late.open_one_batch(here + "/in", named, T, V);
      check(!opened && late.why == EXISTS && picture(here) == as_it_was,
            "a single batch's start leaves a folder with an empty .tmp and no mark.txt",
            std::string(of_a_spool ? "out/ of a spool directory: " : "a folder of its own: ") + late.why);
    }
  }
  { // A single batch that ends well: its response is released, mark.txt and .tmp go, and -clear finds
    // no staging directory.
    const std::string at = new_place(), scene = "a single batch ends well", out = at + "/out7";
    folder(at + "/in");
    const char word[4] = {1, 0, 0, 0};
    bool released;
    {
      Spool one_batch;
      released = one_batch.open_one_batch(at + "/in", out, T, V) &&
                 one_batch.stage_answer("a", {{"y", word, 4}, {"proof", "proof", 5}}) && one_batch.release(-1) &&
                 one_batch.settled(-1);
      if (released) one_batch.tidy();
      // The wipe of an exit that follows must not fail. This does not test the flag staging_is_gone
      // (spool.h): on a local file system the call can return true without it too.
      check(released && one_batch.drop_what_is_staged(), scene,
            "a step before tidy() failed, or drop_what_is_staged() after tidy() returned false");
    }
    const Picture after = picture(out);
    std::string said;
    check(released && after == Picture{"a.proof\tfile\tproof", "a.y\tfile\t" + std::string(word, 4)} &&
              Spool::clear(out, said) == 2 && said == none_there(out) && picture(out) == after,
          scene, said);
  }
  if (geteuid()) { // What the system denies. Modes do not stop root.
    const std::string at = new_place(), scene = "the system refuses", tmp = at + "/spool/out/.tmp/";
    spool_at(at + "/spool"), file(tmp + id(0) + "/proof", "p"), file(tmp + id(1) + "/proof", "p"),
        file(tmp + id(2) + "/proof", "p");
    chmod((tmp + id(0)).c_str(), 0500), chmod((tmp + id(1)).c_str(), 0000);
    std::string said;
    const int status = Spool::clear(at + "/spool", said);
    chmod((tmp + id(0)).c_str(), 0700), chmod((tmp + id(1)).c_str(), 0700);
    check(status == 1 &&
              said == SAYS + "COULD NOT BE REMOVED: " + tmp + id(0) +
                          "/proof: Permission denied.\n" + SAYS + "COULD NOT BE READ: " + tmp +
                          id(1) + ": Permission denied.\n" + may_still_lie(at + "/spool/out/.tmp"),
          scene, said);
    const std::string zero(1, '\0'); // the proof in the directory that cannot be written is itself writable: it is overwritten
    check(picture(at + "/spool/out/.tmp") == Picture{id(0) + "\tfolder", id(0) + "/proof\tfile\t" + zero,
                                                     id(1) + "\tfolder", id(1) + "/proof\tfile\tp"},
          scene, "the two stay, and the third goes");
    {
      chmod((tmp + id(0)).c_str(), 0500);
      Spool s;
      const bool starts = s.open(at + "/spool", T, V);
      chmod((tmp + id(0)).c_str(), 0700);
      check(!starts && strstr(s.why, "Run cp_prove -clear on the folder named with -spool") &&
                picture(tmp + id(0)) == Picture{"proof\tfile\t" + zero},
            scene, std::string("a start over a proof that cannot be removed: ") + s.why);
    }
    chmod((at + "/spool/out").c_str(), 0000);
    said.clear();
    const int shut = Spool::clear(at + "/spool", said);
    chmod((at + "/spool/out").c_str(), 0700);
    const std::string user = std::to_string((unsigned long)geteuid()),
                      ways = "; it would have to belong to user " + user + ", who runs this, with mode 0700. If the folder is yours and only its mode is closed, give "
                             "yourself its permissions back (chmod 700) and run this again. If not, run every step on one folder the same way, with sudo or without: "
                             "what a run as root or under sudo makes belongs to root.\n";
    check(shut == 1 && said == SAYS + "nothing was touched: " + at + "/spool/out cannot be opened: Permission denied. It belongs to user " + user + ", with mode 0000" + ways, scene, said);
    chmod(at.c_str(), 0000); // the folder above the named one
    said.clear();
    const int above = Spool::clear(at + "/spool", said);
    chmod(at.c_str(), 0700);
    check(above == 1 && said == SAYS + "nothing was touched: " + at + "/spool cannot be opened: Permission denied. Its owner and mode cannot be read either (a folder above it may be closed)" + ways,
          scene, said);
    said.clear();
    check(!Spool::clear(at + "/spool", said), scene,
          "with the modes mended the call clears: " + said);
    // A mark.txt that cannot be read, as root's is for an account that comes back without sudo.
    const std::string problem = "its mark.txt cannot be opened: Permission denied (what a run under sudo writes belongs to root: run the "
                                "service and the client both with sudo, or neither)";
    file(tmp + id(0) + "/proof", "p"), out_at(at + "/out7"), file(at + "/out7/.tmp/a/proof", "p");
    chmod((at + "/spool/mark.txt").c_str(), 0000), chmod((at + "/out7/.tmp/mark.txt").c_str(), 0000);
    const Picture before = picture(at);
    std::string clear_says, start_says, out_says;
    const int cleared = Spool::clear(at + "/spool", clear_says), started = start(at + "/spool", start_says),
              out_cleared = Spool::clear(at + "/out7", out_says);
    const bool untouched = picture(at) == before;
    chmod((at + "/spool/mark.txt").c_str(), 0600), chmod((at + "/out7/.tmp/mark.txt").c_str(), 0600);
    check(cleared == 1 && started == 1 && out_cleared == 1 && untouched &&
              clear_says == not_examined(at + "/spool", problem, at + "/spool/out/.tmp") &&
              start_says == cannot_be_used(at + "/spool", problem) &&
              out_says == not_examined(at + "/out7/.tmp", problem, at + "/out7/.tmp"),
          scene, "mark.txt cannot be read:\n" + clear_says + start_says + out_says);
  } else {
    printf("left out, because the test runs as root: the scenes in which the system refuses a "
           "removal or a reading\n");
  }

  { // If a sync fails, the call's status is not 0, and a start fails.
    const std::string at = new_place(), scene = "a sync fails", tmp = at + "/spool/out/.tmp";
    spool_at(at + "/spool"), file(tmp + "/" + id(0) + "/proof", "p"),
        file(tmp + "/" + id(1) + "/notes", "mine");
    struct stat is;
    for (const std::string &one : {tmp, tmp + "/" + id(1)}) {
      std::string said;
      sync_fails_for = stat(one.c_str(), &is) ? 0 : is.st_ino;
      const int status = Spool::clear(at + "/spool", said);
      check(sync_fails_for && status == 1 &&
                said == SAYS + "COULD NOT BE SYNCED TO THE DISK: " + one +
                            ": Input/output error.\n" + may_still_lie(tmp) +
                            left(tmp + "/" + id(1) + "/notes") + not_ours(at + "/spool"),
            scene, said);
    }
    unlink((tmp + "/" + id(1) + "/notes").c_str());
    sync_fails_for = stat(tmp.c_str(), &is) ? 0 : is.st_ino;
    {
      Spool s;
      check(!s.open(at + "/spool", T, V) &&
                strstr(s.why, "Run cp_prove -clear on the folder named with -spool"),
            scene, std::string("a start: ") + s.why);
    }
    sync_fails_for = 0;
    std::string said;
    check(!Spool::clear(at + "/spool", said) && said == clean(tmp), scene,
          "with the sync sound again: " + said);
    // At a first start the sync of the directory fails after mark.txt got its name: the message says
    // that the file is there, and the next start accepts it.
    folder(at + "/new");
    sync_fails_for = stat((at + "/new").c_str(), &is) ? 0 : is.st_ino;
    said.clear();
    int started = start(at + "/new", said);
    const std::string not_synced = "its mark.txt is there, but the folder cannot be synced to the disk: Input/output error";
    check(sync_fails_for && started == 1 && said == cannot_be_used(at + "/new", not_synced) &&
              picture(at + "/new") == Picture{"mark.txt\tfile\t" + mark_bytes("spool")},
          scene, "a first start: " + said);
    sync_fails_for = 0, said.clear(), started = start(at + "/new", said);
    check(!started, scene, "with the sync sound again, a start: " + said);
    // The sync of the temporary file fails: there is no mark.txt, and the temporary file is removed.
    static std::string unused;
    unused = at + "/unused", folder(unused);
    at_the_next_sync = [] {
      struct stat it;
      for (const fs::directory_entry &one : fs::directory_iterator(unused)) sync_fails_for = stat(one.path().c_str(), &it) ? 0 : it.st_ino;
    };
    said.clear(), started = start(unused, said);
    check(sync_fails_for && started == 1 && fs::is_empty(unused) &&
              said == cannot_be_used(unused, "it has no mark.txt yet, and the file cannot be written: Input/output error"),
          scene, "a first start, the file's own sync: " + said);
    sync_fails_for = 0, at_the_next_sync = nullptr;
    // The directory's sync fails at the start of a single batch, whose .tmp exists only from then on.
    static std::string new_tmp;
    new_tmp = at + "/out7/.tmp", folder(at + "/in");
    at_the_next_sync = [] {
      struct stat it;
      sync_fails_for = stat(new_tmp.c_str(), &it) ? 0 : it.st_ino;
    };
    {
      Spool one_batch;
      const bool opened = one_batch.open_one_batch(at + "/in", at + "/out7", T, V);
      check(sync_fails_for && !opened && one_batch.why == "the new folder named with -out cannot be used: " + not_synced +
                                                                ". The folder was created and holds no response: name another" &&
                is_the_mark(new_tmp + "/mark.txt", mark_bytes("out")), scene,
            std::string("cp_prove -out: ") + one_batch.why);
    }
    sync_fails_for = 0, at_the_next_sync = nullptr;
  }
  { // A staged file is overwritten before it is removed: seen through a descriptor that was opened
    // beforehand. For every user, root too.
    const std::string at = new_place(), scene = "overwritten before it is removed",
                      proof = at + "/spool/out/.tmp/" + id(0) + "/proof";
    spool_at(at + "/spool"), file(proof, "secret");
    const int held = open(proof.c_str(), O_RDONLY | O_CLOEXEC);
    std::string said;
    char bytes[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    check(held >= 0 && !Spool::clear(at + "/spool", said) && !fs::exists(proof) &&
              pread(held, bytes, sizeof bytes, 0) == 6 && !memcmp(bytes, "\0\0\0\0\0\0", 6),
          scene, said);
    if (held >= 0) close(held);
  }
  { // A start that fails for a foreign name in work/ or a stop that is not an empty file removes
    // nothing, and names the call; the call then clears what was staged and leaves the rest alone.
    const std::string at = new_place(), scene = "a start refused for work/ or stop",
                      tmp = at + "/spool/out/.tmp";
    for (const char *odd : {"/spool/work/notes", "/spool/stop"}) {
      spool_at(at + "/spool"), file(tmp + "/" + id(0) + "/proof", "p"), file(at + odd, "mine");
      const Picture before = picture(at);
      {
        Spool s;
        check(!s.open(at + "/spool", T, V) && strstr(s.why, "Nothing was removed.") &&
                  strstr(s.why, "run cp_prove -clear on it") && picture(at) == before,
              scene, std::string(odd) + ": " + s.why);
      }
      std::string said;
      check(!Spool::clear(at + "/spool", said) &&
                picture(at) == without(before, {"spool/out/.tmp/" + id(0)}),
            scene,
            std::string(odd) + ": the call clears what was staged, and leaves work/ and stop alone: " + said);
      unlink((at + odd).c_str());
    }
    for (const bool a_link : {false, true}) { // a directory or a link named stop: the same message
      const std::string here = new_place();
      spool_at(here + "/spool");
      if (a_link)
        file(here + "/empty"), link_to("../empty", here + "/spool/stop");
      else
        folder(here + "/spool/stop");
      const Picture before = picture(here);
      Spool s;
      check(!s.open(here + "/spool", T, V) && strstr(s.why, "a stop that is not an empty file") &&
                picture(here) == before,
            scene, std::string(a_link ? "a link" : "a folder") + " named stop: " + s.why);
    }
  }

  { // in, work or out is no directory: a start fails, and nothing is made where a link leads. Only
    // the directories before it in open()'s order (in, work, out) are made, in the spool directory.
    const std::string scene = "a start where in, work or out is no directory";
    const std::vector<std::function<void(const std::string &home)>> states = {
        [](const std::string &home) { folder(home + "/in"), file(home + "/work", "mine"); },
        [](const std::string &home) { folder(home + "/in"), link_to("../theirs", home + "/work"); },
        [](const std::string &home) { link_to("../theirs", home + "/in"), folder(home + "/work"); },
        [](const std::string &home) {
          link_to("../theirs", home + "/out");
        }};
    for (size_t i = 0; i < states.size(); i++) {
      const std::string at = new_place();
      theirs_at(at), file(at + "/mine/mark.txt", mark_bytes("spool")), states[i](at + "/mine");
      const Picture before = picture(at);
      Spool s;
      check(!s.open(at + "/mine", T, V) &&
                without(picture(at), i == 3 ? std::vector<std::string>{"mine/in", "mine/work"} : std::vector<std::string>{}) == before &&
                s.why == at + "/mine/" + (i < 2 ? "work" : i == 2 ? "in" : "out") +
                             " is a link or a file: cp_prove needs a folder of that name and does not follow links. Nothing that was in the spool directory was touched. "
                             "Look at it (ls -ld). If another folder is the service's spool directory, name that one. If not, name a new, empty folder of your own "
                             "(mkdir -m 700): the requests in this one stay there and get no response",
            scene, "state " + std::to_string(i) + ": " + s.why);
    }
  }
  { // A spool directory that a kill or a full disk left incomplete: empty (n = -2), with an
    // unfinished mark.txt (-1), then with mark.txt and each prefix of the five directories, in
    // open()'s order. -clear changes nothing (status 2 or 0); a start completes it.
    const std::string scene = "honest states of a spool directory";
    const char *const made[] = {"/in", "/work", "/out", "/out/.tmp", "/in/.tmp"};
    for (int n = -2; n <= 5; n++) {
      const std::string at = new_place(), spool = at + "/spool";
      folder(spool);
      if (n == -1) file(spool + "/mark.txt.making.0123456789abcdef", "cp-folder 1\nkind sp");
      if (n >= 0) file(spool + "/mark.txt", mark_bytes("spool"));
      for (int k = 0; k < n; k++) folder(spool + made[k]);
      const Picture before = picture(at);
      std::string said;
      check(Spool::clear(spool, said) == (n >= 4 ? 0 : 2) &&
                said == (n >= 4 ? clean(spool + "/out/.tmp") : none_there(spool)) &&
                picture(at) == before,
            scene, "-clear after " + std::to_string(n) + " of the five: " + said);
      Spool s;
      bool all_five = s.open(spool, T, V) && is_the_mark(spool + "/mark.txt", mark_bytes("spool"));
      for (const char *name : made) all_five = all_five && fs::is_directory(spool + name);
      check(all_five, scene, "a start after " + std::to_string(n) + " of the five: " + s.why);
    }
    { // A killed service: a request in in/, one in work/, a released response, staged responses.
      uint32_t words[T + 2] = {1, 2, 3, 4, 5, 6, 7, 8, T - 1, T};
      for (const bool a_start : {false, true}) {
        const std::string at = new_place(), spool = at + "/spool", tmp = spool + "/out/.tmp";
        spool_at(spool), staged_at(tmp, {id(0), id(1), id(2), id(3), id(4)});
        const std::string waits = Spool::lay(spool, words, T), taken = Spool::lay(spool, words, T);
        rename((spool + "/in/" + taken + ".req").c_str(),
               (spool + "/work/" + taken + ".req").c_str());
        for (const char *end : {".y", ".tag", ".proof"})
          file(spool + "/out/" + id(9) + end, "named");
        file(spool + "/stop");
        std::string said;
        Spool s;
        check(a_start ? s.open(spool, T, V) : !Spool::clear(spool, said), scene,
              std::string("a killed service's directory: ") + (a_start ? s.why : said.c_str()));
        bool gone = true;
        for (int i = 0; i < 5; i++) gone = gone && !fs::exists(tmp + "/" + id(i));
        check(gone && fs::exists(spool + "/in/" + waits + ".req") &&
                  fs::exists(spool + "/work/" + taken + ".req") &&
                  fs::exists(spool + "/out/" + id(9) + ".proof"),
              scene, "what was staged is gone; the requests and the released response stay");
      }
    }
    // A spool directory that has lost in/, work/ or both, as after a copy by a tool that drops
    // empty directories: -clear clears it, and a start completes and clears it.
    for (const std::string lost : {"in", "work", "in and work"})
      for (const bool a_start : {false, true}) {
        const std::string at = new_place(), spool = at + "/spool",
                          staged = spool + "/out/.tmp/" + id(0);
        file(staged + "/y", "yyyy"), file(staged + "/proof", "p"), file(spool + "/mark.txt", mark_bytes("spool"));
        if (lost == "work") folder(spool + "/in/.tmp");
        if (lost == "in") folder(spool + "/work");
        std::string said;
        Spool s;
        const bool good = a_start
                              ? s.open(spool, T, V) && fs::is_directory(spool + "/in/.tmp") &&
                                    fs::is_directory(spool + "/work")
                              : !Spool::clear(spool, said) && said == clean(spool + "/out/.tmp");
        check(good && !fs::exists(staged), scene,
              (a_start ? "a start without " : "-clear without ") + lost + ": " +
                  (a_start ? s.why : said.c_str()));
      }
  }
  { // A start follows no link, not even under the named directory's own name.
    const std::string at = new_place();
    folder(at + "/spool"), link_to("spool", at + "/by-a-link");
    Spool s;
    check(!s.open(at + "/by-a-link", T, V) && strstr(s.why, "/by-a-link is a link or a file") && fs::is_empty(at + "/spool"),
          "a start through a link to the caller's own directory is refused, and nothing is made behind it", s.why);
  }

  // mark.txt: first its bytes, which other programs must be able to write and read.
  check(mark_bytes("spool").size() == 99 && mark_bytes("out").size() == 97 && mark_bytes("spool") == Spool::mark_text("spool") &&
            mark_bytes("out") == Spool::mark_text("out") && !strcmp(Spool::MARK_FILE, "mark.txt"),
        "the bytes of mark.txt", Spool::mark_text("spool"));
  { // A spool directory with a staged response and, in turn: cp_prove's mark.txt, a front end's with a
    // commitment ID, and each wrong one.
    const std::string staged = "spool/out/.tmp/" + id(0);
    auto state = [=](const std::function<void(const std::string &at)> &lay_the_mark) {
      return [=](const std::string &at) {
        spool_at(at + "/spool", false), file(at + "/" + staged + "/y", "yyyy"), file(at + "/" + staged + "/proof", "p"), lay_the_mark(at);
      };
    };
    auto cleared = [](const std::string &at) { return clean(at + "/spool/out/.tmp"); };
    auto nothing = [](const std::string &) { return ""; };
    const auto own_mark = state([](const std::string &at) { file(at + "/spool/mark.txt", mark_bytes("spool")); });
    play({"its own mark.txt: -clear", own_mark, "spool", 0, {staged}, cleared}, the_call, false);
    play({"its own mark.txt: a start", own_mark, "spool", 0, {staged}, nothing}, start, false);
    const auto front_ends_mark = state([](const std::string &at) { file(at + "/spool/mark.txt", mark_bytes("spool", A_FRONT_ENDS_ID)); });
    play({"a front end's mark.txt: -clear", front_ends_mark, "spool", 0, {staged}, cleared}, the_call, false);
    play({"a front end's mark.txt: a start", front_ends_mark, "spool", 0, {staged}, nothing}, start, false);
    for (const WrongMark &wrong : wrong_marks("/spool/mark.txt", true)) {
      const std::string problem = wrong.problem;
      play({"mark.txt is wrong: " + wrong.name + ": -clear", state(wrong.lay), "spool", 1, {},
            [=](const std::string &at) { return not_marked(at + "/spool", true, problem); }}, the_call, false);
      // Only a missing mark.txt is excused where nothing is staged.
      play({"mark.txt is wrong: " + wrong.name + ": -clear, and nothing is staged",
            [=](const std::string &at) { spool_at(at + "/spool", false), wrong.lay(at); }, "spool", 1, {},
            [=](const std::string &at) { return not_marked(at + "/spool", true, problem); }}, the_call, false);
      play({"mark.txt is wrong: " + wrong.name + ": a start", state(wrong.lay), "spool", 1, {},
            [=](const std::string &at) { return no_start(at + "/spool", problem); }}, start, false);
    }
  }
  for (const WrongMark &wrong : wrong_marks("/out7/.tmp/mark.txt", false))
    for (const bool staged : {true, false})
      play({"mark.txt of -out is wrong: " + wrong.name + (staged ? ": -clear" : ": -clear, and nothing is staged"),
            [=](const std::string &at) {
              folder(at + "/out7/.tmp"), wrong.lay(at);
              if (staged) file(at + "/out7/.tmp/a/proof", "p");
            },
            "out7", 1, {}, [=](const std::string &at) { return not_marked(at + "/out7", false, wrong.problem); }}, the_call, false);
  for (const Scene &scene : first_starts()) play(scene, start, false);
  for (const Scene &scene : marks_alone()) play(scene, start, false);
  { // A limit: all that a service leaves, with a staged response, but no mark.txt; also with a file
    // of the operator's own at its top. A start and -clear touch nothing, so no call clears it.
    const std::string scene = "a spool directory that has lost mark.txt";
    for (const bool with_notes : {false, true}) {
      const std::string at = new_place(), spool = at + "/spool";
      spool_at(spool, false), file(spool + "/out/.tmp/" + id(0) + "/proof", "p"), file(spool + "/in/" + id(7) + ".req", "waits");
      if (with_notes) file(spool + "/stop"), file(spool + "/notes.txt", "mine");
      const Picture before = picture(at);
      std::string start_says, clear_says;
      const int started = start(spool, start_says), cleared = Spool::clear(spool, clear_says);
      check(started == 1 && cleared == 1 && start_says == no_start(spool, NOT_EMPTY, with_notes ? "in, notes.txt, out, stop, work" : "in, out, work") &&
                clear_says == not_marked(spool, true, NO_MARK) &&
                picture(at) == before,
            scene, start_says + clear_says);
    }
  }
  { // Another program, such as a front end, writes its mark.txt while cp_prove writes its own: here
    // when cp_prove syncs its temporary file. Neither file replaces the other. cp_prove reads what is
    // there, and the start goes on.
    const std::string scene = "two programs mark one folder at the same time";
    static std::string theirs, their_bytes;
    const std::string at = new_place(), spool = at + "/spool";
    folder(spool);
    theirs = spool + "/mark.txt", their_bytes = mark_bytes("spool", A_FRONT_ENDS_ID);
    at_the_next_sync = [] { file(theirs, their_bytes); };
    std::string said;
    const int status = start(spool, said);
    const bool met = !at_the_next_sync;
    at_the_next_sync = nullptr;
    check(met && is_the_mark(theirs, their_bytes) && !status, scene, said);
  }
  { // mark.txt is gone again as soon as cp_prove has written it: here when cp_prove syncs the directory,
    // its second sync. The start fails. The directory is empty, so the message lists no names.
    const std::string at = new_place(), spool = at + "/spool";
    folder(spool);
    static std::string written;
    written = spool + "/mark.txt";
    at_the_next_sync = [] { at_the_next_sync = [] { unlink(written.c_str()); }; };
    std::string said;
    const int status = start(spool, said);
    const bool met = !at_the_next_sync;
    at_the_next_sync = nullptr;
    check(met && status == 1 && said == no_start(spool, NO_MARK) && fs::is_empty(spool), "mark.txt is removed as soon as it is written", said);
  }
  // The controls.
  loud = false;
  auto fails = [&](const Call &stand_in, const char *which, const std::set<std::string> &wanted, const std::vector<Scene> &cases) {
    differing.clear();
    for (const Scene &scene : cases) play(scene, stand_in, false);
    const std::set<std::string> found = differing;
    loud = true;
    for (const std::string &name : wanted)
      check(found.count(name), std::string("control: ") + which,
            "this scene did not fail: " + name);
    loud = false;
  };
  fails(removes_too_much, "a call that removes whatever lies in the staging folder",
        {"another closed folder of the user's with cp_prove's names in out/.tmp and .tmp",
         "another closed folder of the user's with .tmp/a/proof and .tmp/a/y",
         "a spool directory without mark.txt, and a response is staged",
         "foreign names in the staging folder", "foreign names in a response's folder",
         "in a spool a response's name is an id: old/proof stays",
         "foreign names in the folder of -out: no word on -spool",
         ".tmp with mark.txt, an unfinished one and a response: only the response goes",
         "a link under a response's name is left, and not followed", "a folder where a file is due",
         "a file where a folder is due",
         "more than twenty are left",
         ".tmp is a link, beside a staging folder that could be cleared",
         "names in a response's folder that no response holds"},
        all);
  fails(leaves_the_proofs, "a call that leaves the staged proofs and says that nothing is left",
        {"every form, in a spool", "every form, in the folder of -out",
         "in/, work/ and the named responses stay", "-out: released files with names of every form"},
        all);
  fails(asking_for_no_mark(the_call), "a call that asks for no mark.txt",
        {"another closed folder of the user's with cp_prove's names in out/.tmp and .tmp",
         "another closed folder of the user's with .tmp/a/proof and .tmp/a/y",
         "another closed folder of the user's: each name in its .tmp could be a response's",
         "a spool directory without mark.txt, and a response is staged",
         "a spool directory without mark.txt, and nothing is staged",
         "one of two staging folders is not marked: neither is cleared",
         "the spool's out/ is named: its .tmp has no mark.txt of its own",
         "the spool's in/ is named, and in/.tmp contains a file: it is left", "an empty .tmp"},
        all);
  fails(asking_for_no_mark(start), "a start that asks for no mark.txt",
        {"a start: another closed folder of the user's with cp_prove's names in out/.tmp and .tmp",
         "a start: a folder with one file of the user's", "a start: requests wait, and in/ holds another name",
         "a start: requests wait, and the top holds another name", "a start: out holds a name"},
        first_starts());
  std::set<std::string> each;
  for (const Scene &scene : marks_alone()) each.insert(scene.name);
  fails(makes_first, "a start that makes its subfolders before it asks for mark.txt", each, marks_alone());
  loud = true;

  printf("spool_clear_test: %d checks, %d failed\n", checks, differ);
  if (differ) return printf("the work folder stays: %s\n", own.c_str()), 1;
  if (chdir("/")) return 2;
  std::error_code fault;
  fs::remove_all(own, fault); // it follows no link
  if (fault)
    return printf("the work folder could not be removed and stays: %s: %s\n", own.c_str(),
                  fault.message().c_str()),
           2;
  return 0;
}
