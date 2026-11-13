param(
    [string]$Distro = "v7",
    [string]$Root = "/mnt/c/project/v7"
)
# Developer-only WSL scheduling diagnostic (sibling of busy_test.ps1): pins
# guest busy loops to ALL guest CPUs and samples host LP utilization. Run
# from the repo root:
#   powershell -File src\linux\wsl\busy_test_all.ps1 [-Distro v7] [-Root /mnt/c/project/v7]
# The defaults name this development machine's WSL setup; override both for
# any other checkout/distro.
$cpus = 0,1,2,3,4,5
foreach ($c in $cpus) {
  Start-Process wsl.exe -ArgumentList '-d',$Distro,'--','taskset','-c',"$c",'bash',"$(($Root -replace '\\','/'))/src/linux/wsl/burn.sh",'14' -WindowStyle Hidden
}
Start-Sleep -Seconds 4
$samples = @{}
for ($s=0; $s -lt 10; $s++) {
  $rows = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object { $_.Name -match '^\d+$' }
  foreach ($r in $rows) {
    $n=[int]$r.Name
    if (-not $samples.ContainsKey($n)) { $samples[$n]=@() }
    $samples[$n] += [int]$r.PercentProcessorTime
  }
  Start-Sleep -Milliseconds 500
}
wsl.exe -d $Distro -- pkill -f burn.sh | Out-Null
"Guest VMs=6, busy loops pinned to ALL guest CPUs 0-5"
foreach ($k in ($samples.Keys | Sort-Object)) {
  $avg=[math]::Round(($samples[$k] | Measure-Object -Average).Average,0)
  "LP{0}  {1}%" -f $k,$avg
}
