# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The option "reference_directory" of this package's vLLM loader (--model-loader-extra-config).

The option names the directory with the caller's reference copies of a model's small files. The
loader takes the copies' SHA-256 digests here, in memory, and passes them on as the option "sha256"
of integer_model.load_from_folder(). So a caller who has the copies at hand need not put the digests
on a command line or into a file. The module lies in this package because only the vLLM loader takes
the option: load_from_folder() does not know it. It needs no vLLM.
"""

import hashlib
import json
import os
import stat

from integer_model import ParameterError


def replace_reference_directory(options: dict, folder: str | os.PathLike) -> None:
    """Replaces the entry "reference_directory" of the loader's options, if they have one, by the
    entry "sha256" that digests_of_reference_copies() gives for it and the model directory folder.
    The entry "reference_files": 5 beside it demands all five copies, and is taken out too.

    Raises ParameterError if "sha256" or "sha256_file" is given too, if "reference_files" is not 5,
    or if it is given without "reference_directory".
    """
    here = "The options in --model-loader-extra-config"
    all_five = "reference_files" in options
    if all_five and (type(files := options.pop("reference_files")) is not int or files != 5):
        raise ParameterError(f'{here}: "reference_files" must be the number 5, or be left out.')
    if "reference_directory" not in options:
        if all_five:
            raise ParameterError(f'{here}: "reference_files" needs "reference_directory".')
        return
    if "sha256" in options or "sha256_file" in options:
        raise ParameterError(
            f'{here}: give only one of "sha256", "sha256_file" and "reference_directory".'
        )
    options["sha256"] = digests_of_reference_copies(
        options.pop("reference_directory"), folder, all_five
    )


def digests_of_reference_copies(
    path: object, folder: str | os.PathLike, all_five: bool = False
) -> dict[str, str]:
    """Reads the option "reference_directory": the absolute path of a directory, outside the model
    directory folder, that holds the caller's reference copies of the model's small files. Returns
    what the option "sha256" would hold: the SHA-256 of integer_model.json, policy.txt and KL.bin
    there, and of wsums.txt and fold.txt too if all_five is true, if that integer_model.json puts
    the weights in GGUF files, or if they lie there. With five digests, load_from_folder() refuses a
    model whose weights are in no GGUF files. all_five fixes their number beforehand: whether the
    two files lie there at this moment then decides only between five digests and a refusal. The
    digests are taken in memory and written nowhere, so there is no file of digests to keep secret
    or to remove. No other file of the directory is looked at.

    Raises ParameterError if the path is not absolute or leads into folder, or if one of these files
    is missing, cannot be read, is not a regular file, or is the very file that it is to be compared
    with: by device and inode, the file of its name in folder or, for the last two, in the set-up
    directory (a link or a hard link to it, say), since a file always equals itself. Modes and
    owners are not looked at: wsums.txt itself lies there, so keep the directory as closed as the
    weights.
    """
    where = 'The option "reference_directory" (an entry of --model-loader-extra-config)'
    if not isinstance(path, str) or not os.path.isabs(path) or "\0" in path:
        raise ParameterError(f"{where} must be the absolute path of a directory.")
    inside = os.path.realpath(folder)
    if os.path.commonpath([os.path.realpath(path), inside]) == inside:
        raise ParameterError(f"{where}: {path!r} lies in the model directory; keep it outside.")

    def copy_of(name: str, compared_in: str) -> bytes:
        file, other = os.path.join(path, name), os.path.join(compared_in, name)
        try:
            # O_NONBLOCK: a FIFO without a writer must not hold up the load.
            handle = os.open(file, os.O_RDONLY | os.O_NONBLOCK)
            try:
                about = os.fstat(handle)
                if not stat.S_ISREG(about.st_mode):
                    raise ParameterError(f"{where}: {file!r} is not a regular file.")
                with open(handle, "rb", closefd=False) as copy:
                    data = copy.read()
            finally:
                os.close(handle)
        except OSError as error:
            raise ParameterError(f"{where}: cannot read {file!r}: {error.strerror}.") from None
        try:
            same = os.path.samestat(about, os.stat(other))
        except OSError:  # missing or closed: the load itself reports it
            same = False
        if same:
            raise ParameterError(
                f"{where}: {file!r} and {other!r} are one file under two names, so comparing "
                "them would show nothing. Keep each reference copy as a file of its own."
            )
        return data

    copies = {
        name: copy_of(name, inside) for name in ("integer_model.json", "policy.txt", "KL.bin")
    }
    setup_folder = _setup_folder(copies["integer_model.json"])
    for name in ("wsums.txt", "fold.txt"):
        if all_five or setup_folder is not None or os.path.lexists(os.path.join(path, name)):
            copies[name] = copy_of(name, os.path.join(inside, setup_folder or ""))
    return {name: hashlib.sha256(data).hexdigest() for name, data in copies.items()}


def _setup_folder(description: bytes) -> str | None:
    """The entry "setup_folder" of an integer_model.json that puts the weights in GGUF files. None
    for other weights, and where the bytes are no JSON object with that entry as a string.
    """
    try:
        settings = json.loads(description.decode())["weight_settings"]
        found = settings["setup_folder"] if settings["blocks"] == "gguf" else None
    except (ValueError, KeyError, TypeError):
        return None
    return found if isinstance(found, str) else None
