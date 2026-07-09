"""Guarantee a valid path to a SpooledTemporaryFile after rollover has occurred."""

import tempfile
from io import BytesIO, TextIOWrapper


class SpooledNamedTemporaryFile(tempfile.SpooledTemporaryFile):
    """Modified tempfile.SpooledTemporaryFile() to support file names."""

    def rollover(self):
        """Overwrite rollover to use NamedTemporaryFile."""
        if self._rolled:
            return
        file = self._file
        newfile = self._file = tempfile.NamedTemporaryFile(**self._TemporaryFileArgs)  # ty: ignore[unresolved-attribute] # False positive
        del self._TemporaryFileArgs  # ty: ignore[unresolved-attribute]

        pos = file.tell()
        if hasattr(newfile, "buffer"):
            if not isinstance(file, TextIOWrapper):
                raise TypeError(f"Expected a text file, got {type(file)}")
            newfile.buffer.write(file.detach().getvalue())  # ty: ignore[unresolved-attribute] # False positive
        else:
            if not isinstance(file, BytesIO):
                raise TypeError(f"Expected a binary file, got {type(file)}")
            newfile.write(file.getvalue())
        newfile.seek(pos, 0)

        self._rolled = True
