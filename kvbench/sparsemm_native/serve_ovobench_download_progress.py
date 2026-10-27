"""Small read-only status page for the detached OVO-Bench download."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

ROOT = Path('/home/yujun/program/kvbench')
DOWNLOAD = ROOT / 'data/ovobench_upstream_chunks_20261021/status.json'
EXPERIMENT = ROOT / 'results/ovobench_simplestream_upstream_recent4_fps1_20261021/status.json'

PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OVO-Bench 下载进度</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:3rem auto;padding:0 1rem;color:#17212b;background:#f8fafc}
.card{background:white;border:1px solid #dce3ea;border-radius:12px;padding:1.5rem;box-shadow:0 3px 12px #00000008}
h1{font-size:1.5rem;margin-top:0}.big{font-size:2rem;font-weight:700;margin:1rem 0}
progress{width:100%;height:22px}dl{display:grid;grid-template-columns:9rem 1fr;gap:.5rem 1rem}
dt{color:#526171}dd{margin:0}small{color:#526171}#error{color:#a33}
</style><main class="card"><h1>OVO-Bench 官方视频下载</h1>
<div id="amount" class="big">正在读取…</div><progress id="bar" max="100" value="0"></progress>
<dl><dt>进度</dt><dd id="percent">—</dd><dt>最近速度</dt><dd id="speed">—</dd>
<dt>预计剩余</dt><dd id="eta">—</dd><dt>下载状态</dt><dd id="state">—</dd>
<dt>实验进度</dt><dd id="experiment">—</dd><dt>状态更新时间</dt><dd id="updated">—</dd></dl>
<small>每 5 秒自动刷新。预计时间根据最近的下载速度计算，网络波动时会变化。</small>
<p id="error"></p></main><script>
const $=id=>document.getElementById(id);
let previous=null;
const gb=x=>(x/1e9).toFixed(2)+' GB';
async function refresh(){try{
 const response=await fetch('/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);
 const {download:d,experiment:e}=await response.json();
 const amount=d.downloaded_bytes||0,total=d.total_bytes||1,now=Date.now();
 const percent=100*amount/total;
 let speed=d.speed_bytes_per_second||0;
 if(previous&&amount>=previous.amount&&now>previous.now) speed=(amount-previous.amount)*1000/(now-previous.now);
 previous={amount,now};
 $('amount').textContent=gb(amount)+' / '+gb(total);
 $('bar').value=percent;$('percent').textContent=percent.toFixed(2)+'%';
 $('speed').textContent=(speed/1e6).toFixed(2)+' MB/s';
 const hours=speed>0?(total-amount)/speed/3600:Infinity;
 $('eta').textContent=Number.isFinite(hours)?hours.toFixed(1)+' 小时（仅下载）':'暂无法估计';
 $('state').textContent=d.status||'未知';
 $('experiment').textContent=(e.state||'未知')+'；'+(e.completed_queries||0)+' / '+(e.total_queries||3035)+' 题';
 $('updated').textContent=d.updated_at?new Date(d.updated_at).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'}):'—';
 $('error').textContent='';
 }catch(err){$('error').textContent='读取失败：'+err.message}}
refresh();setInterval(refresh,5000);
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
        elif self.path == '/status':
            try:
                body = json.dumps({'download': json.loads(DOWNLOAD.read_text()),
                                   'experiment': json.loads(EXPERIMENT.read_text())}).encode()
            except (OSError, ValueError) as exc:
                self.send_error(503, str(exc))
                return
            content_type = 'application/json; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', 8765), Handler).serve_forever()
