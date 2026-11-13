param(
    [string]$Distro = "v7",
    [string]$Root = "/mnt/c/project/v7"
)
# Developer-only WSL scheduling diagnostic: pins guest busy loops to the
# guest's logical CPUs and samples host logical-processor utilization, to
# observe how WSL spreads guest vCPUs over host LPs. Run from the repo root:
#   powershell -File src\linux\wsl\busy_test.ps1 [-Distro v7] [-Root /mnt/c/project/v7]
# The defaults name this development machine's WSL setup; override both for
# any other checkout/distro.
$cpus = 0,2,4,6,8,10
foreach ($c in $cpus) {
  Start-Process wsl.exe -ArgumentList '-d',$Distro,'--','taskset','-c',"$c",'bash',"$(($Root -replace '\\','/'))/src/linux/wsl/burn.sh",'14' -WindowStyle Hidden
}
Start-Sleep -Seconds 4
$samples = @{}
for ($s=0; $s -lt 8; $s++) {
  $rows = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object { $_.Name -match '^\d+$' }
  foreach ($r in $rows) {
    $n=[int]$r.Name
    if (-not $samples.ContainsKey($n)) { $samples[$n]=@() }
    $samples[$n] += [int]$r.PercentProcessorTime
  }
  Start-Sleep -Milliseconds 500
}
wsl.exe -d $Distro -- pkill -f burn.sh | Out-Null
"Guest busy loops pinned to guest CPUs: 0,2,4,6,8,10"
"Host logical processor utilization:"
foreach ($k in ($samples.Keys | Sort-Object)) {
  $avg=[math]::Round(($samples[$k] | Measure-Object -Average).Average,0)
  "LP{0}  {1}%" -f $k,$avg
}
