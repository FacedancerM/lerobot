from __future__ import annotations

from collections.abc import Iterator

import torch

from lerobot.datasets import EpisodeAwareSampler
from lerobot.scripts import lerobot_train

FROZEN_BATCH_SIZE = 64


class FullBatchEpisodeAwareSampler(EpisodeAwareSampler):
    """EpisodeAwareSampler with deterministic complete batches only.

    The upstream DataLoader uses ``drop_last=False``. Real24 has 3,790 frames,
    so an unmodified sampler would produce a 14-sample optimizer batch once per
    epoch. The matched contract requires a common batch of 64. This sampler
    keeps the upstream epoch permutation and deterministically takes its first
    3,776 positions (59 complete batches), with no duplicate or oversampling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source_num_frames = self._num_frames
        self._num_frames -= self._num_frames % FROZEN_BATCH_SIZE
        if self._num_frames <= 0:
            raise ValueError("Dataset cannot form one complete frozen batch")

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._source_num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._frame_index(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._frame_index(k)


def main() -> None:
    lerobot_train.EpisodeAwareSampler = FullBatchEpisodeAwareSampler
    lerobot_train.main()


if __name__ == "__main__":
    main()

