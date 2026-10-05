param([string]$Name, [int]$Ctx = 196608, [int]$Target = 190000, [string]$Exe = 'D:\llm\build\qc-mtp\bin\llama-server.exe', [string]$Extra = '')
# Live-config server at a larger -c, fills the context to ~$Target tokens, logs GPU shared-memory peak.
$env:LLAMA_MTP_DRAFT_HEAD_Q4 = '1'; $env:LLAMA_MTP_DRAFT_VOCAB = '98304'
$args0 = @('-m','D:\llm\scan\scan-all-q6k.gguf','-ngl','99','-fa','on','-ub','512','-b','2048','-t','8','-np','1',
  '--spec-type','draft-mtp','--spec-draft-n-max','7','--spec-draft-p-min','0.6','-ctk','q8_0','-ctv','q8_0',
  '--temp','0.6','--top-p','0.95','--top-k','20','--min-p','0','-c',"$Ctx",'--cache-ram','4096',
  '--host','127.0.0.1','--port','8099','--chat-template-file','D:\llm\agentbench\templates\official.jinja')
if ($Extra -ne '') { $args0 += ($Extra.Trim() -split ' ') }
$env:PATH = (Split-Path -Parent $Exe) + ';' + $env:PATH
Set-Location (Split-Path -Parent $Exe)
$log = "D:\ninfer\analysis\tune\fill_$Name.log"
$job = Start-Job -ScriptBlock { param($e, $a, $l) pwsh -NoProfile -File D:\ninfer\gpu-run.ps1 model min=25 $e @a *> $l } -ArgumentList $Exe, (,$args0), $log
function GpuGB { ((Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Maximum).Maximum / 1GB }
$ok = $false; for ($i = 0; $i -lt 150; $i++) { Start-Sleep 2; try { if ((Invoke-RestMethod http://127.0.0.1:8099/health -TimeoutSec 2).status -eq 'ok') { $ok = $true; break } } catch {} }
if (-not $ok) { "$Name : server did not start"; Get-Content $log -Tail 5 } else {
  "$Name loaded gpu_shared={0:N2} GB" -f (GpuGB)
  $f = Get-ChildItem D:\llm\src\llama-qc-mtp\src\*.cpp, D:\llm\src\llama-qc-mtp\ggml\src\ggml-opencl\*.cpp | Sort Length -Desc | Select -First 6
  $all = ($f | % { Get-Content $_.FullName -Raw }) -join "`n"; while ($all.Length -lt 900000) { $all += "`n" + $all }
  $n = [int]($Target * 2.1)
  for ($k = 0; $k -lt 8; $k++) { $big = $all.Substring(0, [math]::Min($n, $all.Length)); $tk = (Invoke-RestMethod http://127.0.0.1:8099/tokenize -Method Post -Body (@{content=$big}|ConvertTo-Json) -ContentType 'application/json').tokens.Count; if ([math]::Abs($tk - $Target) -le 2000) { break }; $n = [int]($n * $Target / $tk) }
  "tokens=$tk"
  $peak = 0; $mon = Start-Job { $p = 0; while ($true) { $v = ((Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Maximum).Maximum / 1GB; if ($v -gt $p) { $p = $v; Set-Content D:\ninfer\analysis\tune\fill_peak.txt $p }; Start-Sleep 5 } }
  $body = @{ messages = @(@{ role = 'user'; content = "Summarize what this code does in 3 bullets:`n$big" }); max_tokens = 200 } | ConvertTo-Json -Depth 5
  try { $r = Invoke-RestMethod http://127.0.0.1:8099/v1/chat/completions -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 5400; $t = $r.timings
    "$Name OK prompt_n=$($t.prompt_n) pp=$([math]::Round($t.prompt_per_second,1)) tg=$([math]::Round($t.predicted_per_second,1)) acc=$($t.draft_n_accepted)/$($t.draft_n)" }
  catch { "$Name FAIL: $($_.Exception.Message)"; Get-Content $log -Tail 5 }
  Stop-Job $mon; "peak gpu_shared={0:N2} GB" -f [double](Get-Content D:\ninfer\analysis\tune\fill_peak.txt)
}
Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -ieq $Exe -and ($_.CommandLine -match '8099') } | Stop-Process -Force
Get-Job | Wait-Job -Timeout 60 | Out-Null; Get-Job | Remove-Job -Force
