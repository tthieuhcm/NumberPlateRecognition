import six
from typing import Text, List
import os


def list_directory(path):
    # type: (Text) -> List[Text]
    """Returns all files and folders excluding hidden files.

    If the path points to a file, returns the file. This is a recursive
    implementation returning files in any depth of the path."""

    if not isinstance(path, six.string_types):
        raise ValueError("Resource name must be a string type")

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        results = []
        for base, dirs, files in os.walk(path):
            # remove hidden files
            goodfiles = filter(lambda x: not x.startswith('.'), files)
            results.extend(os.path.join(base, f) for f in goodfiles)
        return results
    else:
        raise ValueError("Could not locate the resource '{}'."
                         "".format(os.path.abspath(path)))


def list_files(folder_path):
    # type: (Text) -> List[Text]
    """Returns all files excluding hidden files.
    If the path points to a file, returns the file."""
    try:
        return [fn for fn in list_directory(folder_path) if os.path.isfile(fn)]
    except:
        return []
