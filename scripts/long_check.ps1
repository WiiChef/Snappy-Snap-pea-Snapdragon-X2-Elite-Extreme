param([string]$Name, [string]$Spec = '', [string]$Exe = 'D:\llm\build\qc-mtp\bin\llama-server.exe', [string]$Ub = '512', [int]$Chars = 180000)
$args0 = @('-m','D:\llm\scan\scan-all-q6k.gguf','-ngl','99','-fa','on','-ub',$Ub,'-b','2048','-t','8','-np','1',
  '--temp','0.6','-c','98304','--host','127.0.0.1','--port','8099','--chat-template-file','D:\llm\agentbench\templates\official.jinja')
if ($Spec -ne '') { $args0 += ($Spec -split ' ') }
$env:PATH = (Split-Path -Parent $Exe) + ';' + $env:PATH
Set-Location (Split-Path -Parent $Exe)
$log = "D:\ninfer\analysis\tune\long_$Name.log"
$job = Start-Job -ScriptBlock { param($e, $a, $l) pwsh -NoProfile -File D:\ninfer\gpu-run.ps1 model min=25 $e @a *> $l } -ArgumentList $Exe, (,$args0), $log
$ok = $false; for ($i = 0; $i -lt 150; $i++) { Start-Sleep 2; try { if ((Invoke-RestMethod http://127.0.0.1:8099/health -TimeoutSec 2).status -eq 'ok') { $ok = $true; break } } catch {} }
if (-not $ok) { "$Name : server did not start" } else {
  $f = Get-ChildItem D:\llm\src\llama-qc-mtp\src\*.cpp | Sort Length -Desc | Select -First 3
  $one = (($f | % { Get-Content $_.FullName -Raw }) -join "`n").Substring(0, 160000); $all = $one + "`n" + $one + "`n" + $one; $n = $Chars
  while ($true) { $big = $all.Substring(0, $n); $tk = (Invoke-RestMethod http://127.0.0.1:8099/tokenize -Method Post -Body (@{content=$big}|ConvertTo-Json) -ContentType 'application/json').tokens.Count; if ($tk -le 88000 -and $tk -ge 80000) { "tokens=$tk chars=$n"; break }; $n = [math]::Min($all.Length, [int]($n * 85000 / $tk)) }
  $body = @{ messages = @(@{ role = 'user'; content = "Summarize what this code does in 3 bullets:`n$big" }); max_tokens = 150 } | ConvertTo-Json -Depth 5
  try { $r = Invoke-RestMethod http://127.0.0.1:8099/v1/chat/completions -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 1200; $t = $r.timings
    "$Name OK prompt_n=$($t.prompt_n) pp=$([math]::Round($t.prompt_per_second,1)) tg=$([math]::Round($t.predicted_per_second,1)) acc=$($t.draft_n_accepted)/$($t.draft_n)" }
  catch { "$Name FAIL: $($_.Exception.Message)"; Get-Content $log -Tail 3 }
}
Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -ieq $Exe } | Stop-Process -Force
Get-Job | Wait-Job -Timeout 60 | Out-Null; Get-Job | Remove-Job -Force
