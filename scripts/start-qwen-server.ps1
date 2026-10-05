# Starts the Qwen3.6-35B-A3B llama-server on 127.0.0.1:8080 if it isn't already listening.
# Used by a local agent wrapper to start the server on demand.
# -Hold keeps this script running while the server is up.
param([switch]$Hold)

$port = 8080
# Qualcomm llama.cpp fork + local branch mtp-smallbatch (D:\llm\src\llama-qc-mtp, build qc-mtp):
# MTP speculative decoding with fixed multi-column/dp4a verify kernels and a 64k-row draft head,
# +39% decode (32.6 -> 45.2 tok/s mean, D:\ninfer\analysis\llama-mtp-results.md). all-Q6_K experts / Q8_0 rest.
# Flash attention + ub512 (ub1024 + the MTP draft context ran out of memory at 96k: FA compile err -6).
# Live check 2026-10-04: chat decode 49 tok/s (temp 0.6), 8k-token prompt prefill 471 tok/s.
# q8_0 KV: with f16 KV the MTP context ran out of host memory at ~30k tokens (ub512) / ~65k (ub256);
# q8_0 survives an 85k prompt, KLD vs f16 within noise (D:\llm\kld\kv-*.log).
# Previous config (stock build, --spec-type ngram-mod): start-qwen-server.ps1.bak-20261004-premtp
$exe = 'D:\llm\build\qc-mtp\bin\llama-server.exe'
$serverArgs = @(
  '-m', 'D:\llm\scan\scan-all-q6k.gguf',
  '-ngl', '99', '-fa', 'on', '-ub', '512', '-b', '2048', '-t', '8', '-np', '1',
  '--spec-type', 'draft-mtp', '--spec-draft-n-max', '7', '--spec-draft-p-min', '0.6',
  '-ctk', 'q8_0', '-ctv', 'q8_0',
  '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0', '--presence-penalty', '0',
  '-c', '98304', '--cache-ram', '4096', '--host', '127.0.0.1', '--port', "$port",
  '--chat-template-file', 'D:\llm\agentbench\templates\official.jinja'
)
$logDir = 'C:\NInfer\logs'
# MTP draft head: Q4_0 copy of the first 96k output rows, drafter only (verified output unchanged).
# Agent bench 2026-10-05: 64k Q8_0 42.6 tok/s (acc 77.8%) -> 96k Q4_0 43.9 (acc 80.5%).
$env:LLAMA_MTP_DRAFT_HEAD_Q4 = '1'
$env:LLAMA_MTP_DRAFT_VOCAB   = '98304'

function Test-Listening { [bool](Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) }

if (-not (Test-Listening)) {
  New-Item -ItemType Directory -Force $logDir | Out-Null
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden `
    -RedirectStandardOutput "$logDir\qwen-server-$stamp.out.log" -RedirectStandardError "$logDir\qwen-server-$stamp.err.log"
}

if ($Hold) {
  while ($true) {
    Start-Sleep -Seconds 30
    if (-not (Test-Listening) -and -not (Get-Process llama-server -ErrorAction SilentlyContinue)) { break }
  }
}
