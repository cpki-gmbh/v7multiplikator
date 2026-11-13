Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class Topo {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool GetLogicalProcessorInformationEx(int rel, byte[] buf, ref uint len);
}
"@
$len = 0
[Topo]::GetLogicalProcessorInformationEx(0, $null, [ref]$len) | Out-Null
$buf = New-Object byte[] $len
if (-not [Topo]::GetLogicalProcessorInformationEx(0, $buf, [ref]$len)) { throw "call failed $([Runtime.InteropServices.Marshal]::GetLastWin32Error())" }
$off = 0
"Relationship=ProcessorCore records (relationship,size,groupCount,coreMask):"
while ($off -lt $len) {
  $rel  = [BitConverter]::ToUInt32($buf, $off)
  $size = [BitConverter]::ToUInt32($buf, $off+4)
  $gc   = [BitConverter]::ToUInt16($buf, $off+8+22)
  $mask = [BitConverter]::ToUInt64($buf, $off+8+24)
  "rel=$rel size=$size groupCount=$gc mask=0x$($mask.ToString('X'))"
  $off += $size
}
"--- Win32_Processor ---"
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,ThreadCount | Format-List
