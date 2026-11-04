# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
#
# make OUT=<folder> test   (OUT is needed with every target)
# make test            builds and runs every <folder>/tests/*.cpp, tests/*.cpp and tests/hiding/*.cpp for every setting in SETTINGS
# make TESTS="a/tests/x.cpp .." test   the same, for the named test sources only (paths from this folder)
# make long            runs the exact tests of hiding that know the argument "long" with it: they then run all their cases, some of which "make test" leaves out for time (key_exact_test, which tests nothing without the argument, gets it from "make test" too)
# make toy             builds toy/*.cpp
# make programs        builds every other <folder>/*.cpp (programs that are no tests, and the four sources in tests/, which are tests)
# make all             test, toy and programs; make without a target does the same
# make SETTINGS=f2h1   names the default and only setting (f = field 2, h = hash 1)
#
# What a test prints is kept beside its target, in <target>.said, and printed after its name: with -j the lines of several may mingle
# on the screen; the files do not (../tests/new/run_new.py reads them).
# A test is a program that prints one line and returns 0.
CXX      ?= g++
# -fno-ipa-modref: on every g++ call, a precaution against a fault of g++ 13.3.0: an object handed over by reference could lose, silently,
# the writes that fill it. The fault was seen in another program built at -O1 -fno-inline, never at plain -O2, and never in this
# folder. A compiler that does not know the flag: make CXXFLAGS="-std=c++17 -O2 -Wall -Wextra".
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -fno-ipa-modref
SETTINGS ?= f2h1
# Where programs are built: a folder of your own OUTSIDE the source tree, so that the tree holds sources alone.
ifeq ($(strip $(OUT)),)
$(error say where to build: make OUT=/some/folder/of/your/own ...)
endif
ifneq ($(filter $(abspath $(CURDIR))/%,$(abspath $(OUT))/),)
$(error OUT lies inside this folder: build outside the source tree)
endif
# The rules below run the programs that are in OUT, so no other user may be able to put a file there. An OUT that does not exist is
# created here, with its missing parents, with access for its owner only. After that, OUT must be a directory, not a symbolic
# link, that the caller owns and that neither its group nor other users can write. find takes type, owner and mode from one
# lstat call, and in a sticky directory such as /tmp nobody else can rename or remove what the caller owns. The directories
# above OUT, and what is already below it, are not examined. abspath removes "..", "." and a final slash, so that find looks
# at the last part of the name itself and not through a link.
override OUT := $(abspath $(OUT))
OUT_IS_SAFE := $(shell (umask 077 && mkdir -p '$(dir $(OUT))' && mkdir '$(OUT)') 2> /dev/null; \
  find '$(OUT)' -maxdepth 0 -type d -uid "$$(id -u)" ! -perm /022 2> /dev/null | grep -q . && echo yes)
ifneq ($(OUT_IS_SAFE),yes)
$(error OUT=$(OUT) cannot be used. The programs that are built there are run, so it must be a directory that make can create, or a directory (not a symbolic link) that you own and that neither its group nor other users can write. For example: B=$$(mktemp -d), then OUT=$$B)
endif

LONG   := $(patsubst %,tests/hiding/%_exact_test.cpp,key lin lkp member rng toy)
TESTS  := $(sort $(wildcard */tests/*.cpp) $(wildcard tests/*.cpp) $(wildcard tests/hiding/*.cpp))
# A test runs without arguments, but for this one, which tests nothing without its argument (about nine minutes, 5.4 GB).
ARGS_tests/hiding/key_exact_test := long
TOYS   := $(sort $(wildcard toy/*.cpp))
PROGRAMS := $(filter-out $(TOYS),$(sort $(wildcard */*.cpp)))
HEADERS := $(wildcard *.h */*.h */tests/*.h tests/hiding/*.h)

flags = -DBLK_FIELD=$(patsubst f%,%,$(firstword $(subst h, ,$(1)))) -DBLK_HASH=$(lastword $(subst h, ,$(1)))

define per_setting
$(OUT)/$(1)/%.bin: %.cpp $(HEADERS)
	@mkdir -p $$(dir $$@)
	$(CXX) $(CXXFLAGS) $(call flags,$(1)) -I . $$< -o $$@
$(OUT)/$(1)/%.ran: $(OUT)/$(1)/%.bin
	@$$< $$(ARGS_$$*) > $$(@:.ran=.said) 2>&1; status=$$$$?; printf '%s %s: %s\n' $(1) $$* "$$$$(cat $$(@:.ran=.said))"; [ $$$$status -eq 0 ] && touch $$@
$(OUT)/$(1)/%.ran-long: $(OUT)/$(1)/%.bin
	@$$< long > $$(@:.ran-long=.said-long) 2>&1; status=$$$$?; printf '%s %s long: %s\n' $(1) $$* "$$$$(cat $$(@:.ran-long=.said-long))"; [ $$$$status -eq 0 ] && touch $$@
endef
$(foreach s,$(SETTINGS),$(eval $(call per_setting,$(s))))

.PHONY: test long toy programs all
all: test toy programs
test: $(foreach s,$(SETTINGS),$(TESTS:%.cpp=$(OUT)/$(s)/%.ran))
long: $(foreach s,$(SETTINGS),$(LONG:%.cpp=$(OUT)/$(s)/%.ran-long))
toy: $(foreach s,$(SETTINGS),$(TOYS:%.cpp=$(OUT)/$(s)/%.bin))
programs: $(foreach s,$(SETTINGS),$(PROGRAMS:%.cpp=$(OUT)/$(s)/%.bin))
.SECONDARY:
