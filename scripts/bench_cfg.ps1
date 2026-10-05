param([string]$Name, [string]$Spec = '', [string]$Exe = 'D:\llm\build\qc-mtp\bin\llama-server.exe')
# Live-server settings (C:\NInfer\start-qwen-server.ps1) on port 8099 with a given speculative config.
# Each prompt runs twice with fixed seeds (1, 2); prints decode tok/s and draft acceptance.
$args0 = @('-m','D:\llm\scan\scan-all-q6k.gguf','-ngl','99','-fa','on','-ub','512','-b','2048','-t','8','-np','1',
  '--temp','0.6','--top-p','0.95','--top-k','20','--min-p','0','--presence-penalty','0',
  '-c','98304','--host','127.0.0.1','--port','8099','--chat-template-file','D:\llm\agentbench\templates\official.jinja')
if ($Spec -ne '') { $args0 += ($Spec -split ' ') }
Set-Location (Split-Path -Parent $Exe)
$job = Start-Job -ScriptBlock { param($e, $a) pwsh -NoProfile -File D:\ninfer\gpu-run.ps1 model min=25 $e @a 2>&1 | Out-Null } -ArgumentList $Exe, (,$args0)
$ok = $false; for ($i = 0; $i -lt 150; $i++) { Start-Sleep 2; try { if ((Invoke-RestMethod http://127.0.0.1:8099/health -TimeoutSec 2).status -eq 'ok') { $ok = $true; break } } catch {} }
if (-not $ok) { "$Name : server did not start"; Get-Job | Stop-Job; exit 1 }
$prompts = Get-Content D:\ninfer\analysis\tune\prompts.json -Raw | ConvertFrom-Json
$warm = @{ messages = @(@{ role = 'user'; content = 'Say hi.' }); max_tokens = 16 } | ConvertTo-Json -Depth 5
$null = Invoke-RestMethod http://127.0.0.1:8099/v1/chat/completions -Method Post -Body $warm -ContentType 'application/json' -TimeoutSec 300
$rows = @()
foreach ($p in $prompts) {
  foreach ($seed in 1, 2) {
    $body = @{ messages = $p.messages; max_tokens = 400; seed = $seed; cache_prompt = $false } | ConvertTo-Json -Depth 6
    $r = Invoke-RestMethod http://127.0.0.1:8099/v1/chat/completions -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 900
    $t = $r.timings
    $rows += [pscustomobject]@{ cfg = $Name; prompt = $p.name; seed = $seed; n = $t.predicted_n; tgs = [math]::Round($t.predicted_per_second, 2);
      pp = [math]::Round($t.prompt_per_second, 1); pn = $t.prompt_n; acc = "$($t.draft_n_accepted)/$($t.draft_n)" }
  }
}
Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -ieq $Exe } | Stop-Process -Force
Get-Job | Wait-Job -Timeout 60 | Out-Null; Get-Job | Remove-Job -Force
$rows | Export-Csv -Append -NoTypeInformation D:\ninfer\analysis\tune\results.csv
$rows | Format-Table -AutoSize | Out-String
