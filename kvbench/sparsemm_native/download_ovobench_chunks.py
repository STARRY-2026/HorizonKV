"""Download the byte-identical official clips required by upstream SimpleStream."""
import concurrent.futures
import fcntl
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import tarfile
import threading

SOURCE = Path('/home/yujun/program/hermes/data/ovobench/official_20261014')
ROOT = Path('/home/yujun/program/kvbench/data/ovobench_upstream_chunks_20261021')
spec = importlib.util.spec_from_file_location('checked_ovo_download', SOURCE / 'download.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((SOURCE / 'download_manifest.json').read_text())
    tree = json.loads((SOURCE / 'metadata/hf_tree.json').read_text())
    files = [dict(name=x['path'], size=x['size'], sha256=x['lfs']['oid'],
                  url=f"https://huggingface.co/datasets/JoeLeelyf/OVO-Bench/resolve/{previous['hf_revision']}/{x['path']}")
             for x in tree if x['path'].startswith('chunked_videos.tar.part')]
    files.sort(key=lambda x: x['name'])
    if len(files) != 15:
        raise ValueError('Unexpected official archive count')
    manifest = dict(previous, files=files, total_bytes=sum(x['size'] for x in files),
                    archive_root='chunked_videos', scope='official pre-cut query clips, original archive bytes')
    target = ROOT / 'download_manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Download manifest changed')
    base.atomic_json(target, manifest)
    (ROOT / 'annotations').mkdir(exist_ok=True)
    annotation = ROOT / previous['annotation_path']
    if not annotation.exists():
        annotation.symlink_to(SOURCE / previous['annotation_path'])
    return manifest


class ChunkDownload(base.DownloadJob):
    def verify_clips(self, directory):
        annotations = json.loads((ROOT / self.manifest['annotation_path']).read_text())
        names = []
        for anno in annotations:
            if anno['task'] in ('REC', 'SSR', 'CRR'):
                names.extend(f"{anno['id']}_{index}.mp4" for index in range(len(anno['test_info'])))
            else:
                names.append(f"{anno['id']}.mp4")
        if len(names) != 3035 or len(set(names)) != len(names):
            raise ValueError('Official annotation/query identity mismatch')
        missing = [n for n in names if not (directory / n).is_file() or (directory / n).stat().st_size < 100]
        if missing:
            raise ValueError(f'Missing official query clips: {missing[:10]} ({len(missing)} total)')
        return dict(annotation_records=len(annotations), query_clips=len(names),
                    extracted_clips=sum(1 for _ in directory.glob('*.mp4')))

    def extract(self):
        destination = ROOT / 'chunked_videos'
        if destination.exists():
            return self.verify_clips(destination)
        stage = ROOT / '.extracting'
        stage.mkdir(exist_ok=True)
        self.publish(status='extracting')
        count = 0
        with base.ConcatReader([self.archives / x['name'] for x in self.files]) as reader:
            with tarfile.open(fileobj=reader, mode='r|') as archive:
                for member in archive:
                    self.check_stop()
                    name = PurePosixPath(member.name)
                    if (name.is_absolute() or '..' in name.parts or not name.parts or
                            name.parts[0] != 'chunked_videos' or not (member.isdir() or member.isfile())):
                        raise ValueError(f'Unsafe archive member: {member.name}')
                    archive.extract(member, path=stage, filter='data')
                    count += int(member.isfile())
                    if count % 100 == 0:
                        self.publish(extracted_clips=count)
        summary = self.verify_clips(stage / 'chunked_videos')
        os.replace(stage / 'chunked_videos', destination)
        stage.rmdir()
        return summary

    def run(self):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        with (ROOT / 'download.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            heartbeat = threading.Thread(target=self.heartbeat, daemon=True)
            heartbeat.start()
            try:
                self.publish(status='downloading', workers=6)
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                    futures = [pool.submit(self.download_one, item) for item in self.files]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()
                        except BaseException:
                            self.cancel()
                            raise
                summary = self.extract()
                base.atomic_json(ROOT / 'completion.json', dict(summary, completed_at=base.utc(),
                                 hf_revision=self.manifest['hf_revision']))
                self.publish(status='completed', summary=summary)
                return 0
            except base.Paused as exc:
                self.publish(status='paused', error=str(exc))
                return 130
            except BaseException as exc:
                self.publish(status='error', error=repr(exc))
                raise
            finally:
                self.heartbeat_stop.set()
                heartbeat.join(timeout=6)


if __name__ == '__main__':
    prepare()
    os.environ['OVOBENCH_DOWNLOAD_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['OVOBENCH_DIRECT'] = '1'
    sys.exit(ChunkDownload(ROOT).run())
