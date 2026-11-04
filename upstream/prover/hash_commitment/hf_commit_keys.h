// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The keys that the committer program cp_commit derives from its seed, the provider's 32 secret
// bytes (schedule: prover/common/coin_keys.h). In identifiers `first_table` is the anchor (first
// weight table: all weights as rows with masks, encoded, under a Merkle tree), `copy` a leaf table
// (another table of the same weights with its own masks), `entry` a leaf file (`ivk/leaf/<k>.bin`).
// Hiding rests on this: the same masks, salts or pads must never occur under two Merkle roots. So
// each table's key depends on bytes that are drawn fresh for that creation: the seed for the
// anchor, 32 bytes from the operating system ("system bytes") for a leaf table. `Make` is built
// for this rule; the comment there names two options that give it up and says how weakly the rule
// is enforced.
// Include from prover/hash_commitment/hf_commit.cu only, inside its namespace, after the_seed() and
// lock_in_memory(); that file also includes the headers that this one needs.
#pragma once

// The public inputs of a table's key, known before the table exists: the configuration digest (a
// hash of the public configuration files) and the table's shape words.
struct TableFacts {
  blk::Digest configuration{};
  coin_keys::Shape shape;
  TableFacts() = default;
  // Parses a name that blk::index_name() (blocks/iop/foldeq.h) builds, here with a zero root, so
  // that the shape words are defined in one place. They follow the 64 bytes of the name's two
  // digests, the root and the configuration digest.
  TableFacts(const blk::IndexShape &table, const blk::Digest &c) : configuration(c) {
    const std::vector<blk::u8> name = blk::index_name(blk::Digest{}, table, c);
    blk::Reader words(name.data(), name.size()); // by the format's own reader
    blk::Digest passed_over;
    BLK_REQUIRE(words.get(passed_over) && words.get(passed_over), "a stored table's name begins with two digests");
    for (blk::u64 word; !words.at_end() && words.get_u64(word);) shape.push_back(word);
    BLK_REQUIRE(words.at_end() && !shape.empty(), "behind its two digests a stored table's name is words of shape");
  }
};
static int take_the_seed(const Opt &O); // hf_commit.cu, as the next two
static bool the_coins_key_is(const uint8_t key[32]);
static void hold_the_coins_key(const uint8_t key[32]);

// A weight table's key, in the base class's `w`, and a second copy, `again`, for the self-checks of
// hf_mask_own.cuh and hf_pads.cuh. The table's masks (random values encoded with each row), its
// tree leaves' salts and its rows' pads are derived from the key. It serves to
// recompute columns for openings; `make_the_table` and `make_the_record` (hf_commit.cu) do not take
// it. make_again_the_first_table() fills one for an anchor that already exists.
struct TableKey : hf_key32 {
  uint8_t again[32];
  TableKey() : hf_key32{} {
    memset(again, 0, sizeof again);
    lock_in_memory(this, sizeof *this);
  }
  void wipe() {
    seed_wipe(*this);
    explicit_bzero(again, sizeof again);
  }

private:
  friend struct Make;
  friend void make_again_the_first_table(TableKey &, const TableFacts &);
  // Derives the key from the seed in memory, and `again` from the seed's file, read again: two
  // derivations from memory would agree even if the seed there had been wiped or never filled.
  // Exits with code 4 if the file cannot be read. kept: null for the anchor; for leaf table k the
  // system bytes of this creation, which are in no file yet, so both copies take them from memory.
  void derive(const TableFacts &facts, uint32_t k, const uint8_t *kept) {
    auto from = [&](uint8_t out[32], const uint8_t seed_bytes[32]) {
      blk::CoinKey seed = blk::coin_key_from_bytes(seed_bytes);
      blk::CoinKey key = kept ? coin_keys::copy_table_key(seed, k, kept, facts.configuration, facts.shape) : coin_keys::first_table_key(seed, facts.configuration, facts.shape);
      coin_keys::key_bytes(out, key);
      blk::wipe(seed), blk::wipe(key);
    };
    uint8_t seed[32], key[32];
    the_seed().bytes(seed);
    from(key, seed);
    memcpy(w, key, 32);
    hf_key32 anew{};
    const bool read = seed_from_file(the_seed().file.c_str(), anew);
    memcpy(seed, anew.w, 32);
    from(again, seed);
    seed_wipe(anew);
    explicit_bzero(seed, sizeof seed), explicit_bzero(key, sizeof key);
    if (read) return;
    fprintf(stderr, "cp_commit: internal error (hc-key-file): %s cannot be read a second time\n", the_seed().file.c_str());
    exit(4);
  }
};
// The key type taken by `make_the_table`, which builds a table's tree, and `make_the_record`, which
// computes the table's record (stored answers to samples of its rows). Only `Make` constructs one.
class MakersKey {
public:
  const TableKey &key;
  MakersKey(const MakersKey &) = delete;
  MakersKey &operator=(const MakersKey &) = delete;

private:
  friend struct Make;
  explicit MakersKey(const TableKey &k) : key(k) {}
};

// The system bytes for the key of the leaf table being created. They stay here until the verifier
// program cp_verify has accepted the leaf file; cp_commit then writes them into the secret key
// directory, as `isk/leaf/<k>/fresh.bin`, from which, with the seed, the prover program cp_prove
// derives the key again. A creation that ends earlier loses them; the next has other masks (unless
// `repeatable`: see Make::a_copy()).
struct KeptBytes {
  KeptBytes() {
    memset(bytes_, 0, sizeof bytes_);
    lock_in_memory(this, sizeof *this);
  }
  void wipe() { explicit_bzero(bytes_, sizeof bytes_); }
  const uint8_t *to_be_written() const { return bytes_; } // for the write described above, only
  static constexpr size_t BYTES = 32;

private:
  friend struct Make;
  uint8_t bytes_[32];
};

// Creates tables. It draws each creation's secret and derives the key itself: no argument can
// supply a key or system bytes. Two arguments weaken this. `O` can replace the seed's draw: a test
// build reads the seed from the file of `-test-seed` (`O.test_seed`), and a production build
// derives it from the file of `-public-weights` (`O.public_seed`; prover/common/public_weights.h).
// In any build `repeatable` = true leaves the system bytes zero (see a_copy()). Only `Make` can
// construct the MakersKey that make_the_table() and make_the_record() take. `out` receives the
// TableKey, with which the proofs in the key file (`ivk/key.bin`: the anchor's root and one-time
// proofs about it) and in a leaf file open the new table. The `derivation_check` calls
// (prover/hash_commitment/hf_commit_derivation_check.cuh) give the same inputs to a second
// implementation of the key schedule (prover/second_writing/), which derives the table's key
// itself; the table's masks and a sample of its salts are compared with what it derives. A table is
// built in one or more steps, each of which generates all its masks again; only the first step's
// masks are compared.
// The compiler enforces only these types: make_table() and PartMaker
// (prover/hash_commitment/hf_commit_table.cuh), below make_the_table(), take a plain key, and
// hf_side_record() (prover/hash_commitment/hf_open_side.cuh), below make_the_record(), takes
// whatever pads it is given.
// A source check that is not part of this repository checks that make_table() and hf_side_record()
// have one caller each outside tests, that only `Make` draws the secrets and that
// prover/hash_commitment/hf_commit.cu passes `repeatable` as true only under `-test-seed` or
// `-public-weights`, by patterns and by digests of this file's code from TableKey on. The check is
// weak: by patterns alone it missed 21 of 21 deliberate evasions, and a macro or a function pointer
// evades it.
struct Make {
  // Creates the anchor. seed_is_written() writes the seed's file, from which each key's second copy
  // is derived, and returns 0 or an exit code. tree(MakersKey) builds the tree and the record.
  // Returns 0, or the exit code from drawing or writing the seed.
  template <class Written, class Tree>
  static int the_first_table(const Opt &O, const TableFacts &facts, TableKey &out, Written seed_is_written, Tree tree) {
    if (int rc = take_the_seed(O)) return rc; // the only call that draws the seed
    if (int rc = seed_is_written()) return rc;
    ::derivation_check::the_seed_is(the_seed().w, facts.configuration.w, facts.shape.data(), facts.shape.size());
    out.derive(facts, 0, nullptr);
    ::derivation_check::the_anchors_table_begins();
    tree(MakersKey(out));
    return 0;
  }
  // Creates leaf table k. repeatable: the run has `-test-seed` or `-public-weights`; the system
  // bytes stay zero, and nothing is hidden.
  template <class Tree> static void a_copy(uint32_t k, bool repeatable, const TableFacts &facts, KeptBytes &kept, TableKey &out, Tree tree) {
    kept.wipe();
    if (!repeatable) cp_fresh32(kept.bytes_); // their only draw; exits with code 2 on failure
    out.derive(facts, k, kept.bytes_);
    ::derivation_check::a_leafs_table_begins(k, kept.bytes_);
    tree(MakersKey(out));
  }
};
// For cp_commit's second form, a call that adds a leaf table to an existing commitment: derives the
// anchor's key from the seed that read_the_seed() has read from its file. cp_commit rejects that
// form (`cp-commit-first-form-alone`), so no run reaches this function.
inline void make_again_the_first_table(TableKey &out, const TableFacts &facts) {
  ::derivation_check::the_seed_is(the_seed().w, facts.configuration.w, facts.shape.data(), facts.shape.size());
  out.derive(facts, 0, nullptr);
}

// ---- the process keys: one for the key file, one for each leaf file ----
// Under a process key cp_commit draws one file's private random values (`coin` in identifiers).
// The key file's system bytes stay in memory while that file is created, for hold_the_key_files().
// [keep-for-the-audit] The system bytes of a process key are written to a file only under
// `-keep-for-the-audit`, for one run of tools/audit, which recomputes the private random values of
// `key.bin` and the leaf files and compares. cp_commit accepts the option only with synthetic
// weights (`cp-commit-kept-on-a-model`). Together with the seed the bytes reveal every mask in
// those files. Each file of system bytes is written once, as a new file, right after the draw and
// before a key is derived from the bytes. As far as names show, no source in prover/ outside test
// directories reads one (a source check that is not part of this repository looks for that), and no
// run can continue from a dead run's files: cp_commit requires two new directories. A run that
// fails by a return, exit(), abort(), an uncaught exception or a caught signal (SIGABRT, SIGINT,
// SIGTERM, SIGHUP or SIGPIPE) removes the files, which are on the cleanup list: take_away()
// (prover/hash_commitment/hf_commit_making.h) overwrites each with zeros (limits: see zeros_first()
// there) and then unlinks it. A removal can fail, which cp_commit reports. Not reported: if another
// thread starts the cleanup while the main thread is inside making::write_new(), the file can
// appear after the cleanup and stay (see stand_still() there). SIGKILL or another signal that
// leave_at_a_signal() (same file) does not catch, a crash or a power cut leaves the files in
// `isk/audit/`, with the seed's file in the same `isk`, whose name is the `-isk` argument plus
// `.making` until the run's last renames; nothing is printed then.
// All this is cp_commit's share of the option's conditions; the operator's share is listed in
// docs/operators-duties.md, "The audit with the key".
struct RunKeys {
  // kept_as: the file's full path, or empty to keep nothing. Exits with code 4, before the bytes
  // are used, unless the file is created new, written in full and synced with its directory.
  static void keep(const std::string &kept_as, const uint8_t fresh[32]) {
    if (kept_as.empty() || making::write_new(kept_as, fresh, 32)) return;
    fprintf(stderr, "cp_commit: a making's 32 bytes cannot be kept for the audit (-keep-for-the-audit). Nothing was made from them. The run ends and keeps nothing of isk or ivk. Mend what the line above names, and run cp_commit again\n");
    exit(4);
  }
  // Installs the process key under which the key file's private random values are drawn, derived
  // from the seed, fresh system bytes and the commitment ID (a hash over the configuration digest,
  // the anchor's name, its seeds and answers). repeatable: see Make::a_copy();
  // without it a second creation of the key file would have other random values and would again
  // count opened columns against the anchor's limit (the columns that one table may show in all).
  // kept_as: see keep().
  static void take_the_key_files(const blk::Digest &id, bool repeatable, const std::string &kept_as) {
    uint8_t *fresh = bytes().data;
    explicit_bzero(fresh, 32);
    if (!repeatable) cp_fresh32(fresh);
    keep(kept_as, fresh);
    uint8_t seed[32], key[32];
    the_seed().bytes(seed);
    seed[0] ^= (uint8_t)(cert_doors::once_key_fault() == cert_doors::KEY_OF_ANOTHER_SEED); // a test's injected fault
    derive_key_files(key, seed, fresh, id);
    ::hz::key32_renew(key);
    hold_the_coins_key(key);
    explicit_bzero(seed, sizeof seed), explicit_bzero(key, sizeof key);
  }
  // [own-test-key] Exits with code 4 unless the seed's file can be read again and
  // the_coins_key_is() accepts the key derived from it. That function compares one 32-bit word
  // derived from each key, so another key passes about once in 2^32. Called right after
  // take_the_key_files(), and again when the key file is complete, before a byte of it is used. The
  // check detects a seed in memory that changed between the writing of its file and
  // take_the_key_files(), and an installed key that changed later. A fault in the derivation itself
  // is left to the second implementation of the schedule (namespace `derivation_check`), with which the
  // key is compared here too. Nothing detects a wrong ID or wrong system bytes: every derivation
  // gets the same ones.
  static void hold_the_key_files(const blk::Digest &id) {
    uint8_t seed[32], key[32];
    hf_key32 anew{};
    const bool read = seed_from_file(the_seed().file.c_str(), anew);
    memcpy(seed, anew.w, 32);
    derive_key_files(key, seed, bytes().data, id);
    seed_wipe(anew);
    explicit_bzero(seed, sizeof seed);
    const bool same = read && the_coins_key_is(key);
    if (same) { // [own-test-second-text]: and the second implementation derives the same key
      uint8_t id_bytes[32];
      blk::digest_to_bytes(id, id_bytes);
      ::derivation_check::the_key_files_key_is(key, bytes().data, id_bytes);
    }
    explicit_bzero(key, sizeof key);
    if (same) return;
    fprintf(stderr,
            read ? "cp_commit: internal error (own-test-key): the key under which the key file's random values are drawn is not the key that %s gives. The program or the "
                   "machine has a fault, or the file was replaced during this call. The run ends and keeps nothing of isk or ivk. Run cp_commit again\n"
                 : "cp_commit: internal error (hc-key-file): %s can no longer be read as a regular file of 32 bytes with permissions for its owner only. The run ends and "
                   "keeps nothing of isk or ivk. Run cp_commit again\n",
            the_seed().file.c_str());
    exit(4);
  }
  // Call when the key file's last message is complete, before a byte goes to cp_verify or a file.
  // Exits with code 4 unless salts of the anchor were compared with the second implementation.
  static void the_key_file_is_made() {
    ::derivation_check::the_key_file_ends();
    explicit_bzero(bytes().data, 32);
  }
  // Installs the process key for one creation of leaf file k, derived from the seed, fresh system
  // bytes (stored only under `-keep-for-the-audit`), the commitment ID, and leaf table k's name
  // (blk::index_name()) and answers (of its record, the stored answers to samples of its rows).
  static void take_an_entrys(uint32_t k, const blk::Digest &id, const coin_keys::Bytes &name, const coin_keys::Bytes &answers, bool repeatable, const std::string &kept_as) {
    uint8_t seed_bytes[32], key[32], fresh[32] = {0};
    if (!repeatable) cp_fresh32(fresh);
    keep(kept_as, fresh);
    the_seed().bytes(seed_bytes);
    blk::CoinKey seed = blk::coin_key_from_bytes(seed_bytes), K = coin_keys::entry_key(seed, k, fresh, id, name, answers);
    coin_keys::key_bytes(key, K);
    uint8_t id_bytes[32];
    blk::digest_to_bytes(id, id_bytes);
    ::derivation_check::an_entrys_key_is(key, k, fresh, id_bytes, name, answers); // [own-test-second-text], before a coin is drawn
    ::hz::key32_renew(key);
    hold_the_coins_key(key);
    blk::wipe(seed), blk::wipe(K);
    explicit_bzero(seed_bytes, sizeof seed_bytes), explicit_bzero(key, sizeof key), explicit_bzero(fresh, sizeof fresh);
  }
  // Call when a leaf file's last message is complete, before a byte goes to cp_verify or a file.
  // Exits with code 4 unless salts of the anchor and of the leaf table were compared likewise.
  static void an_entry_is_made() { ::derivation_check::an_entry_ends(); }
  static void wipe() { explicit_bzero(bytes().data, 32); }

private:
  struct Locked {
    uint8_t data[32];
    Locked() {
      memset(data, 0, sizeof data);
      lock_in_memory(this, sizeof *this);
    }
  };
  static Locked &bytes() {
    static Locked &b = *new Locked; // never destroyed, so that the exit handler finds it
    return b;
  }
  static void derive_key_files(uint8_t out[32], const uint8_t seed_bytes[32], const uint8_t fresh[32], const blk::Digest &id) {
    blk::CoinKey seed = blk::coin_key_from_bytes(seed_bytes), K = coin_keys::key_file_key(seed, fresh, id);
    coin_keys::key_bytes(out, K);
    blk::wipe(seed), blk::wipe(K);
  }
};
