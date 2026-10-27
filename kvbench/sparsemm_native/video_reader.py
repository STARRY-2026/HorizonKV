"""Dependency-light copy of KVBench's batch-scoped Decord reader.

The class matches ``sparsemm_protokv.models.qwen2_5_vl_stream`` and avoids
importing the unrelated lmms-eval plugin during OVO inference.
"""


def _close_decord_reader(reader):
    try:
        reader.__del__()
    finally:
        reader._handle = None


class _CopiedVideoBatch:
    def __init__(self, array):
        self.array = array

    def asnumpy(self):
        return self.array


class BatchScopedVideoReader:
    def __init__(self, video_path, num_threads=1):
        from decord import VideoReader
        self.video_path = video_path
        self.num_threads = num_threads
        reader = VideoReader(video_path, num_threads=num_threads)
        try:
            self.frame_count = len(reader)
            self.avg_fps = reader.get_avg_fps()
        finally:
            _close_decord_reader(reader)

    def __len__(self):
        return self.frame_count

    def get_avg_fps(self):
        return self.avg_fps

    def get_batch(self, indices):
        from decord import VideoReader
        reader = VideoReader(self.video_path, num_threads=self.num_threads)
        try:
            array = reader.get_batch(indices).asnumpy()
        finally:
            _close_decord_reader(reader)
        return _CopiedVideoBatch(array)
