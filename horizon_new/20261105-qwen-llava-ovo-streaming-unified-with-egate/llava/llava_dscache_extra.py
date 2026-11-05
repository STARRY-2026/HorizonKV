"""LLaVA 1-fps ordinary 196-token input for four baseline cache policies."""
import contextlib
import io
import torch


def install(Adapter, method):
    original_compress = Adapter.predict_and_compress

    @torch.inference_mode()
    def encode(self, frames, is_last_chunk=False):
        if len(frames) != 1:
            raise RuntimeError('LLaVA input requires one frame per update')
        pix = self.processor.preprocess(frames.numpy(), return_tensors='pt')['pixel_values']
        pix = pix.to(device=self.model.device, dtype=next(self.model.parameters()).dtype)
        if tuple(pix.shape[-2:]) != (384, 384):
            raise RuntimeError('Expected LLaVA 384x384 image')
        features = self.model.encode_images(pix)
        ordinary = self.model.get_2dPool(features, stride=2)
        if ordinary.shape[1] != 196:
            raise RuntimeError('Expected 196 ordinary visual tokens')
        stream = self.stream
        if method in ('sparsemm', 'lukv'):
            stream.metadata = []
            stream.forward(ordinary.reshape(1, -1, ordinary.shape[-1]), compress=True)
        else:
            stream.forward(ordinary.reshape(1, -1, ordinary.shape[-1]), compress=False)
            stream.seen += ordinary.shape[0] * ordinary.shape[1]
            if method == 'fullkv':
                expected = stream.prefix + stream.seen
                if any(any(length != expected for length in layer)
                       for layer in stream.cache.head_lengths()):
                    raise RuntimeError('Full KV lost ordinary visual tokens')
        self.chunks += 1
        if self.chunks == 1:
            print('INPUT_PROFILE ordinary_tokens=196 latest_replay=false', flush=True)

    if method == 'hermes':
        def quiet_compress(self):
            with contextlib.redirect_stdout(io.StringIO()):
                return original_compress(self)
        Adapter.predict_and_compress = quiet_compress
    Adapter.encode_video_chunk = encode
