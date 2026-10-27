"""Resume official OVO archive parts without curl's destructive internal retries.

Each connection writes a separate segment. Only a verified HTTP 206 byte range
is appended to the durable .incomplete prefix, including after a network error.
The existing downloader's manifest, archive hashes, lock, and extraction remain
the source of truth.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from sparsemm_native.download_ovobench_chunks import ChunkDownload, ROOT, base, prepare


def commit_segment(partial: Path, segment: Path, headers: Path, expected: int) -> int:
    """Append a persisted segment iff the final HTTP response matches the prefix."""
    if not segment.exists() or segment.stat().st_size == 0:
        segment.unlink(missing_ok=True)
        headers.unlink(missing_ok=True)
        return 0
    if not headers.exists():
        raise RuntimeError(f'Missing range headers; retained {segment}')
    blocks = re.split(rb'\r?\n\r?\n', headers.read_bytes())
    responses = [block for block in blocks if block.startswith(b'HTTP/')]
    if not responses:
        raise RuntimeError(f'Missing HTTP response; retained {segment}')
    final = responses[-1]
    status = re.match(rb'HTTP/\S+\s+(\d+)', final)
    matches = re.findall(rb'(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$', final)
    if not status or int(status.group(1)) != 206 or len(matches) != 1:
        raise RuntimeError(f'Range response invalid; retained {segment} and {headers}')
    start, end, total = map(int, matches[0])
    offset = partial.stat().st_size if partial.exists() else 0
    size = segment.stat().st_size
    if start != offset or total != expected or end < start or size > end - start + 1:
        raise RuntimeError(f'Range mismatch at {offset}; retained {segment} and {headers}')
    with partial.open('ab') as destination, segment.open('rb') as source:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    segment.unlink()
    headers.unlink()
    return size


class ResumableChunkDownload(ChunkDownload):
    def progress(self):
        amount, sizes = super().progress()
        for item in self.files:
            name = item['name']
            final = self.archives / name
            segment = self.archives / (name + '.segment')
            if not final.exists() and segment.exists():
                extra = min(segment.stat().st_size, item['size'] - sizes[name])
                sizes[name] += extra
                amount += extra
        return amount, sizes

    def download_one(self, item):
        name = item['name']
        final = self.archives / name
        partial = self.archives / (name + '.incomplete')
        segment = self.archives / (name + '.segment')
        headers = self.archives / (name + '.headers')
        expected = item['size']
        url = self.download_urls[name]
        attempt = 0
        while True:
            commit_segment(partial, segment, headers, expected)
            self.check_stop()
            if final.exists():
                if final.stat().st_size != expected or base.sha256(final) != item['sha256']:
                    raise RuntimeError(f'Existing archive failed SHA256: {final}')
                print(f'{base.utc()} Verified {name}', flush=True)
                return
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                raise RuntimeError(f'Partial archive exceeds manifest size: {partial}')
            if offset == expected:
                if base.sha256(partial) != item['sha256']:
                    raise RuntimeError(f'Partial archive failed SHA256; retained {partial}')
                self.check_stop()
                os.replace(partial, final)
                print(f'{base.utc()} Verified {name}', flush=True)
                return

            attempt += 1
            command = ['curl', '--fail', '--location', '--noproxy', '*', '--ipv4',
                       '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '60',
                       '--range', f'{offset}-{expected - 1}', '--dump-header', str(headers),
                       '--output', str(segment), url]
            print(f'{base.utc()} {name}: attempt {attempt}, resume at {offset}/{expected}', flush=True)
            with (self.root / 'logs' / (name + '.resume.log')).open('ab') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log)
                with self.mutex:
                    self.children.add(process)
                try:
                    while process.poll() is None:
                        if self.stop.wait(0.25):
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            break
                finally:
                    with self.mutex:
                        self.children.discard(process)
            committed = commit_segment(partial, segment, headers, expected)
            self.check_stop()
            if process.returncode != 0 or committed == 0:
                print(f'{base.utc()} {name}: curl {process.returncode}, retained {committed} bytes', flush=True)
                if self.stop.wait(min(30, 2 ** min(attempt, 5))):
                    self.check_stop()


if __name__ == '__main__':
    prepare()
    os.environ['OVOBENCH_DOWNLOAD_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['OVOBENCH_DIRECT'] = '1'
    sys.exit(ResumableChunkDownload(ROOT).run())
