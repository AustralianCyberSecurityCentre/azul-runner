"""Guarantee a valid path to a SpooledTemporaryFile after rollover has occurred."""

import tempfile


class SpooledNamedTemporaryFile(tempfile.SpooledTemporaryFile):
    """Modified tempfile.SpooledTemporaryFile() to support file names."""

    def rollover(self):
        """Overwrite rollover to use NamedTemporaryFile."""
        if self._rolled:
            return
        file = self._file
        newfile = self._file = tempfile.NamedTemporaryFile(**self._TemporaryFileArgs)
        del self._TemporaryFileArgs

        pos = file.tell()
        if hasattr(newfile, "buffer"):
            newfile.buffer.write(file.detach().getvalue())
        else:
            newfile.write(file.getvalue())
        newfile.seek(pos, 0)

        self._rolled = True
