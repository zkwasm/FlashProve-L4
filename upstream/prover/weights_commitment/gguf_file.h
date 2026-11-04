// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Reads the GGUF version 3 files of one model (its shards, one or more) for the programs cp_commit
// and cp_prove: parses each shard's metadata and tensor table, and reads the raw bytes of a tensor
// on request. open_folder(), at() and read() do not return on failure: they call stop().
//
// Public source: GGUF is the ggml project's file format (github.com/ggml-org/llama.cpp; MIT licence). Its layout is
// described at the head of ggml/include/gguf.h. This reader follows that description and the numbers of gguf-py's
// gguf/constants.py; it has not been compared with ggml's or gguf-py's readers.
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "common/end_name.h"

namespace gguf {

// The storage types that are read. A tensor of any other type has bytes() == 0, and the code that asks for it refuses
// it. Public source for the numbers, and for the sizes in bytes_of(): the package gguf (gguf-py of the llama.cpp
// project; MIT licence), version 0.19.0, gguf/constants.py: GGMLQuantizationType (F32 = 0, F16 = 1, Q8_0 = 8,
// Q4_K = 12, Q5_K = 13, Q6_K = 14, BF16 = 30) and GGML_QUANT_SIZES (a block's elements and bytes: 32 and 2 + 32 for
// Q8_0; 1 and 2 for F16 and BF16).
enum Type : uint32_t { F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, BF16 = 30 };

// Returns the bytes that 'cells' elements of a type take, or 0 for a type that is not read. 'cells' must be a whole
// number of the type's blocks: 256 elements for Q4_K, Q5_K and Q6_K, 32 for Q8_0, 1 for the others.
inline uint64_t bytes_of(uint32_t type, uint64_t cells) {
  switch (type) {
  case F32:
    return cells * 4;
  case F16:
  case BF16:
    return cells * 2;
  case Q8_0:
    return cells / 32 * 34;
  case Q4_K:
    return cells / 256 * 144;
  case Q5_K:
    return cells / 256 * 176;
  case Q6_K:
    return cells / 256 * 210;
  default:
    return 0;
  }
}

// One entry of a shard's tensor table.
struct Tensor {
  int shard = 0;
  uint64_t offset = 0; // from the start of the shard's file
  int ndims = 0;
  uint64_t dims[4] = {1, 1, 1, 1}; // dims[0] is the contiguous axis
  uint32_t type = 0;
  uint64_t cells() const { return dims[0] * dims[1] * dims[2] * dims[3]; }
  // Q4_K, Q5_K and Q6_K store blocks of 256 elements, Q8_0 blocks of 32 (bytes_of()).
  uint64_t bytes() const { return bytes_of(type, cells()); }
};

// A path for a printed line, with a backslash and each byte that is not printable ASCII as \x and
// two hex digits. A path can end in any file name that the weights' directory holds; shown so, the
// name can neither begin a printed line nor send a control sequence to a terminal.
inline std::string shown(const std::string &path) {
  std::string all;
  for (const unsigned char c : path) {
    char hex[5];
    snprintf(hex, sizeof hex, "\\x%02x", c);
    if (c >= 0x20 && c < 0x7f && c != '\\')
      all += (char)c;
    else
      all += hex;
  }
  return all;
}

// Prints the reason and calls exit(2), which cp_prove's exit handler end_on_exit()
// (prover/prover/ends.h) turns into exit code 1 if nothing else has failed. Callers pass as `where`
// a path, a number or, in at(), the name that at()'s caller asked for, not text read from a shard.
[[noreturn]] inline void stop(const char *what, const std::string &where) {
  name_the_end("cp-weights-files");
  fprintf(stderr, "gguf: %s (%s)\n", what, shown(where).c_str());
  exit(2);
}

// The three hyperparameters come from the first shard's metadata. No destructor: the descriptors
// stay open (all close-on-exec).
class Model {
public:
  std::map<std::string, Tensor> tensors;
  uint64_t embedding_length = 0, block_count = 0, key_length = 128;

  // Opens the model in a directory (`folder` in identifiers and messages): its *.gguf files, sorted
  // by name. Call it once per Model. The directory is listed, and every shard and side file opened,
  // through a single open() of it, so all files of one Model are looked up in one directory even if
  // the path later leads elsewhere. Every build follows a symbolic link at the end of the path.
  //
  // A production build (SERVING_BUILD) opens no file of the directory through a symbolic link, and
  // rejects a file that it sees with a second name (a hard link). This narrows what someone can do
  // who can choose the weights directory but cannot read the weights. A hard link to the weights is one way to have
  // them read beside their own files; a weight file moved there, or made visible there by a bind mount, would pass too:
  // the checks do not ask whose the directory or the file is (docs/limits-all-rows.tsv, row H21.2: reasoned, not run).
  // They are not a reliable barrier (see file_in_the_folder()). fs.protected_hardlinks = 1, or no right to create a
  // name on the weights' file system, stops that link alone, and by row HW-209 only if set on every host that mounts
  // it, before the weights are put there; not for their owner, CAP_FOWNER or an account that can read and write them.
  void open_folder(const std::string &folder) {
    name_ = folder;
    folder_ = ::open(folder.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    const int twin = folder_ < 0 ? -1 : fcntl(folder_, F_DUPFD_CLOEXEC, 0);
    DIR *const list = twin < 0 ? nullptr : fdopendir(twin);
    if (!list) stop("the weights' folder cannot be opened or listed", folder);
    std::vector<std::string> shards;
    while (const struct dirent *const entry = readdir(list)) {
      const std::string name = entry->d_name;
      if (name.size() > 5 && !name.compare(name.size() - 5, 5, ".gguf")) shards.push_back(name);
    }
    closedir(list); // closes only the duplicate
    std::sort(shards.begin(), shards.end());
    if (shards.empty()) stop("the weights' folder holds no file *.gguf", folder);
    for (size_t s = 0; s < shards.size(); s++) read_shard((int)s, shards[s]);
    if (!embedding_length || !block_count) stop("embedding_length or block_count is missing", folder + "/" + shards[0]);
  }
  // Opens `name` in the opened directory. Returns a descriptor, or a NotTaken value with errno set.
  enum NotTaken { CANNOT_BE_OPENED = -1, A_LINK = -2, NOT_REGULAR = -3, A_SECOND_NAME = -4 };
  int file_in_the_folder(const std::string &name, struct stat &is) const {
    const int fd = openat(folder_, name.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | LINKS); // O_NONBLOCK: opening a FIFO does not block
    if (fd < 0) return LINKS && errno == ELOOP ? A_LINK : CANNOT_BE_OPENED;
    int why = fstat(fd, &is) ? CANNOT_BE_OPENED : !S_ISREG(is.st_mode) ? NOT_REGULAR : LINKS && is.st_nlink != 1 ? A_SECOND_NAME : 0;
    // The name must still lead to the opened file, and that file must have one name. This may
    // detect a second name that was removed between openat() and fstat(), but not reliably: a hard
    // link that another process creates and removes in a tight loop while the files are opened can
    // pass both checks. It did so in 182 of 604 starts of a production `cp_prove` of other sources
    // (one trial, at small size, on synthetic weights; two machines, xfs, Linux 6.12; one account
    // owned program and files, so the attacker's means to create the link were assumed;
    // docs/limits-all-rows.tsv, row H21.2). No bound on the rate is known.
    struct stat named;
    if (LINKS && !why && (fstatat(folder_, name.c_str(), &named, AT_SYMLINK_NOFOLLOW) || named.st_dev != is.st_dev || named.st_ino != is.st_ino || named.st_nlink != 1))
      why = A_SECOND_NAME;
    if (why) close(fd), errno = EINVAL;
    return why ? why : fd;
  }
  // Opens a side file of the model (wsums.txt, fold.txt) from the same directory. The caller closes
  // it. Returns nullptr, with errno set, on failure.
  FILE *side(const char *name) const {
    struct stat is;
    const int fd = file_in_the_folder(name, is);
    return fd < 0 ? nullptr : fdopen(fd, "r");
  }
  const Tensor &at(const std::string &name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) stop("a tensor is missing", name);
    return it->second;
  }
  bool has(const std::string &name) const { return tensors.count(name) != 0; }
  // For check_the_digests() in prover/hash_commitment/public_weights_markers.h, which hashes whole
  // shards: the number of shards, and shard s's name in the directory and its descriptor, the one
  // that read() uses. The caller must not close it. s < shard_count() is not checked.
  size_t shard_count() const { return fds_.size(); }
  const std::string &shard_name(size_t s) const { return names_[s]; }
  int shard_descriptor(size_t s) const { return fds_[s]; }
  // Reads bytes [first, first + n) of a tensor. Does not compare first + n with t.bytes(), so it
  // can read past the tensor's end.
  void read(const Tensor &t, uint64_t first, uint64_t n, void *to) const {
    char *p = (char *)to;
    uint64_t at = t.offset + first;
    while (n) {
      const ssize_t got = pread(fds_[t.shard], p, n, (off_t)at);
      if (got <= 0)
        stop("a tensor extends beyond the end of its file, or a read of it failed", std::to_string(t.shard));
      p += got, at += (uint64_t)got, n -= (uint64_t)got;
    }
  }
  // Asks the system to drop a tensor's bytes from its page cache. cp_commit and cp_prove call it once
  // they hold the tensor in memory of their own, so that a large model need not be in memory twice
  // while it loads. The next reader of these bytes, in any process, may have to read them from disk.
  void drop_from_page_cache(const Tensor &t) const {
    drop_from_page_cache(shard_descriptor((size_t)t.shard), t.offset, t.bytes());
  }
  // The same for bytes [first, first + n) of an open file. It is advice: a page only partly in the
  // range stays; on some file systems so does a larger unit of the cache that is only partly in it,
  // and a read that ends just before the range can bring its first bytes back by read-ahead. A
  // failure costs speed and memory only, so the result is ignored. n == 0 asks for nothing:
  // posix_fadvise() takes a length of 0 to mean the rest of the file. Public source: posix_fadvise(2),
  // POSIX_FADV_DONTNEED, https://man7.org/linux/man-pages/man2/posix_fadvise.2.html.
  static void drop_from_page_cache(int fd, uint64_t first, uint64_t n) {
    if (n) (void)posix_fadvise(fd, (off_t)first, (off_t)n, POSIX_FADV_DONTNEED);
  }

private:
  // A flag for openat(); if nonzero, it also enables the hard-link checks.
#ifdef SERVING_BUILD
  static constexpr int LINKS = O_NOFOLLOW;
#else
  static constexpr int LINKS = 0;
#endif
  int folder_ = -1;  // the opened directory
  std::string name_; // the directory's path; used only in messages
  std::vector<int> fds_;           // the shards' descriptors
  std::vector<std::string> names_; // the shards' names in the directory

  struct Cursor {
    FILE *f;
    const std::string &name;
    uint64_t size; // the file's length in bytes
    template <class T> T get() {
      T v;
      if (fread(&v, sizeof v, 1, f) != 1) stop("the header is truncated", name);
      return v;
    }
    std::string str() {
      const uint64_t n = get<uint64_t>();
      if (n > (1u << 20)) stop("a string is too long", name);
      std::string s(n, 0);
      if (n && fread(&s[0], 1, n, f) != n) stop("the header is truncated", name);
      return s;
    }
    // Steps over n bytes; calls stop() if they do not all lie inside the file.
    void skip(uint64_t n) {
      const off_t at = ftello(f);
      if (at < 0 || (uint64_t)at > size || n > size - (uint64_t)at || fseeko(f, (off_t)n, SEEK_CUR))
        stop("the header is malformed: an array cannot be skipped", name);
    }
  };
  // Bytes per GGUF metadata value type. Types 0, 2, 4 and 10 are the unsigned integers; types 8
  // (string) and 9 (array) have no fixed size.
  static uint64_t scalar_size(uint32_t type, const std::string &name) {
    static const int size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type > 12 || !size[type]) stop("an unknown metadata type", name);
    return (uint64_t)size[type];
  }

  // Parses shard s and adds its tensors to `tensors`; calls stop() unless each is inside the file
  // and no two overlap. The header is parsed through a duplicate of the descriptor that read()
  // uses, so the tensor table and the tensors' bytes come from one file, even if the name is given
  // to another file meanwhile. This does not stop someone who can write into the file while it is
  // read.
  void read_shard(int s, const std::string &name) {
    const std::string path = name_ + "/" + name; // used only in messages
    struct stat is;
    const int fd = file_in_the_folder(name, is);
    if (fd == A_LINK) stop("a shard is a symbolic link, which a serving build does not follow: lay the file itself into the folder", path);
    if (fd == A_SECOND_NAME) stop("a shard has a second name (a hard link), which a serving build does not take: lay a file of its own into the folder", path);
    if (fd == NOT_REGULAR) stop("a shard is no regular file", path);
    if (fd < 0) stop("a shard cannot be opened", path);
    const int twin = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    FILE *f = twin < 0 ? nullptr : fdopen(twin, "rb");
    if (!f) stop("a shard cannot be opened", path);
    Cursor c{f, path, (uint64_t)is.st_size};
    if (c.get<uint32_t>() != 0x46554747u) stop("no GGUF magic", path);
    if (c.get<uint32_t>() != 3) stop("the GGUF version is not 3", path);
    const uint64_t ntensors = c.get<uint64_t>(), nkv = c.get<uint64_t>();
    // An entry of the tensor table takes at least 32 bytes, and a metadata entry at least 13.
    if (ntensors > (uint64_t)is.st_size / 32 || nkv > (uint64_t)is.st_size / 13) stop("the header names more entries than the file has room for", path);
    uint64_t alignment = 32;
    std::string arch;
    std::map<std::string, uint64_t> uints;
    for (uint64_t i = 0; i < nkv; i++) {
      const std::string key = c.str();
      const uint32_t type = c.get<uint32_t>();
      if (type == 8) {
        const std::string v = c.str();
        if (key == "general.architecture") arch = v;
      } else if (type == 9) {
        const uint32_t et = c.get<uint32_t>();
        const uint64_t n = c.get<uint64_t>();
        if (et == 8)
          for (uint64_t q = 0; q < n; q++) c.str();
        else {
          if (n >> 59) stop("the header is malformed: an array has 2^59 or more elements", path); // the product cannot wrap
          c.skip(n * scalar_size(et, path));
        }
      } else {
        const uint64_t size = scalar_size(type, path);
        uint64_t v = 0;
        if (fread(&v, 1, size, f) != size) stop("the header is truncated", path);
        if (type == 0 || type == 2 || type == 4 || type == 10) uints[key] = v;
      }
    }
    if (uints.count("general.alignment")) alignment = uints["general.alignment"];
    if (!alignment || (alignment & (alignment - 1))) stop("general.alignment is not a power of two", path);
    if (s == 0) {
      if (arch.empty()) stop("general.architecture is missing", path);
      embedding_length = uints.count(arch + ".embedding_length") ? uints[arch + ".embedding_length"] : 0;
      block_count = uints.count(arch + ".block_count") ? uints[arch + ".block_count"] : 0;
      if (uints.count(arch + ".attention.key_length")) key_length = uints[arch + ".attention.key_length"];
    }
    // A message about an entry of the tensor table gives the entry's number, from 1, and never its
    // name. A name can hold control characters, and after a damaged header the bytes that are read
    // as a name can be bytes of the weights.
    const auto entry = [&path](uint64_t i) { return path + ", entry " + std::to_string(i + 1) + " of the tensor table"; };
    std::vector<std::pair<std::string, Tensor>> table;
    for (uint64_t i = 0; i < ntensors; i++) {
      const std::string name = c.str();
      Tensor t;
      t.shard = s;
      t.ndims = (int)c.get<uint32_t>();
      if (t.ndims < 1 || t.ndims > 4) stop("n_dims is not in the range 1 to 4", entry(i));
      // The shape comes from an untrusted file. Keeping the element count below 2^61 stops cells()
      // and bytes() (at most 4 bytes per element) from overflowing uint64_t. The limit of 2^30 on a
      // dimension is that of tensor() in prover/weights_blocks/gguf_source.cuh.
      uint64_t cells = 1;
      for (int q = 0; q < t.ndims; q++) {
        if (!(t.dims[q] = c.get<uint64_t>())) stop("a dimension is 0", entry(i));
        if (t.dims[q] > (1u << 30)) stop("a dimension is larger than 2^30", entry(i));
        if (cells > (UINT64_MAX >> 3) / t.dims[q]) stop("the tensor has 2^61 or more elements", entry(i));
        cells *= t.dims[q];
      }
      t.type = c.get<uint32_t>();
      // These checks make the divisions in bytes_of() exact; otherwise a buffer sized by bytes() would
      // be a block short.
      if ((t.type == Q4_K || t.type == Q5_K || t.type == Q6_K) && t.dims[0] % 256)
        stop("the rows of a quantized tensor are not a multiple of 256 elements long", entry(i));
      if (t.type == Q8_0 && t.dims[0] % 32)
        stop("the rows of a Q8_0 tensor are not a multiple of 32 elements long", entry(i));
      t.offset = c.get<uint64_t>();
      if (t.offset % alignment) stop("an offset is not a multiple of the alignment", entry(i));
      if (t.offset >> 62) stop("an offset is 2^62 or more", entry(i)); // so that no sum below wraps
      table.push_back({name, t});
    }
    const uint64_t end = (uint64_t)ftello(f), data = (end + alignment - 1) / alignment * alignment;
    fclose(f); // closes only the duplicate; fd stays open
    for (size_t i = 0; i < table.size(); i++) {
      auto &e = table[i];
      e.second.offset += data;
      // If the file shrinks after is.st_size was read, read() calls stop() at the new end.
      if (e.second.bytes() > (uint64_t)is.st_size || e.second.offset > (uint64_t)is.st_size - e.second.bytes())
        stop("a tensor extends beyond the end of its file", entry(i));
      if (!tensors.insert(e).second) stop("a tensor name occurs twice", entry(i));
      // From here on `first` holds the entry's index, for the overlap message; the name is in `tensors`.
      e.first = std::to_string(i);
    }
    // In this overlap check and the bounds check above, a tensor of a type that bytes() does not
    // know counts as empty.
    std::sort(table.begin(), table.end(), [](const auto &a, const auto &b) { return a.second.offset < b.second.offset; });
    for (size_t i = 1; i < table.size(); i++)
      if (table[i - 1].second.offset + table[i - 1].second.bytes() > table[i].second.offset) stop("two tensors overlap", entry(std::stoull(table[i].first)));
    fds_.push_back(fd);
    names_.push_back(name);
  }
};

} // namespace gguf
