param(
    [string]$Distro = "v7",
    [string]$Root = "/mnt/c/project/v7"
)
# Developer-only WSL scheduling diagnostic: runs a long v7 inference inside
# WSL and samples host logical-processor utilization while it burns, to see
# how the worker pool's guest threads land on host LPs. Run from the repo
# root:
#   powershell -File src\linux\wsl\host_usage.ps1 [-Distro v7] [-Root /mnt/c/project/v7]
# The defaults name this development machine's WSL setup; override both for
# any other checkout/distro.
$rootU = $Root -replace '\\','/'
Start-Process wsl.exe -ArgumentList '-d',$Distro,'--','bash','-lc',"cd $rootU && ./bin-linux/inference_example ./bin-linux/libv7_threading.so ./weights/databases/qwen3_0.6b/lz4ht.db -q8 -think 0 -n 4000 -i `"write a very long poem about horse, really make it very long!`" >/tmp/hostusage.log 2>&1" -WindowStyle Hidden
Start-Sleep -Seconds 15
$samples = @{}
for ($s=0; $s -lt 5; $s++) {
  $rows = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object { $_.Name -match '^\d+$' }
  foreach ($r in $rows) {
    if (-not $samples.ContainsKey([int]$r.Name)) { $samples[[int]$r.Name] = @() }
    $samples[[int]$r.Name] += [int]$r.PercentProcessorTime
  }
  Start-Sleep -Milliseconds 700
}
wsl.exe -d $Distro -- bash -c 'pkill -9 -f inference_example' | Out-Null
"LP  avg%"
foreach ($k in ($samples.Keys | Sort-Object)) {
  $avg = [math]::Round(($samples[$k] | Measure-Object -Average).Average,0)
  "LP{0}  {1}%" -f $k, $avg
}
