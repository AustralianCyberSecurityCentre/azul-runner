"""Specialized plugin for downloading files on request from a remote source."""

import typing

from azul_bedrock import models_network as azm

from azul_runner import settings
from azul_runner.models import DownloadJob, Job
from azul_runner.plugin import Plugin
from azul_runner.pusher import Pusher


class DownloadPlugin(Plugin):
    """A specialized plugin used for downloading from."""

    def __init__(self, config: settings.Settings | dict | None = None) -> None:
        super().__init__(config)

        if config is None:
            config = {}
        elif isinstance(config, settings.Settings):
            config = config.model_dump()
        self.pusher = Pusher(self.__class__, config)
        self.network = self.pusher._network

    def upload_sourced_file(self, raw_file: typing.BinaryIO, filename: str = ""):
        """Upload a file and the associated events to Azul."""
        self.pusher.source_downloaded_file_once(
            content=raw_file,
            source_label=self._download_job.source.name,
            references=self._download_job.source.references,
            security=self.SECURITY,
            filename=filename,
        )
        self._is_download_completed = True

    def main_loop(self, limit=-1):
        """Main loop for running a downloader plugin."""
        count = 0
        while limit == -1:
            if limit > 0 and count >= limit:
                break
            count += 1
            try:
                download_job = self.network.fetch_download_job()
                self._download_job = download_job
                if self.network.api.has_binary(
                    source=download_job.source.name, label=azm.DataLabel.CONTENT, sha256=download_job.entity.hash
                ):
                    self.network._notify_download(download_job, azm.DownloadAction.SkippedAlreadyPresent)
                    continue
            except Exception as e:
                raise Exception("Failed to fetch download job!") from e

            try:
                self._is_download_completed = False
                download_action = self.execute_download(DownloadJob(event=download_job))
                if download_action is None:
                    if self._is_download_completed:
                        download_action = azm.DownloadAction.FailedNotFound
                    else:
                        download_action = azm.DownloadAction.Success

                elif download_action == azm.DownloadAction.Requested:
                    raise Exception("Download plugin cannot request to do additional downloads.")

                self.network._notify_download(download_job, action=download_action)
            except Exception as e:
                self.network._notify_download(download_job, azm.DownloadAction.Failed)
                raise e

    def execute_download(self, job: DownloadJob) -> azm.DownloadAction | None:
        """Entrypoint for the download execution.

        Users are expected to use `job.event.entity.hash` to download load the hash.
        Also to upload the file found via download use the command upload_sourced_file
        """
        raise NotImplementedError

    def execute(self, job: Job):
        """Entrypoint for the download execution."""
        pass
