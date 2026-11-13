# PowerShell 5.1 cross-language benchmark -- same 4 workloads, warmup + N reps, median.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File bench_fib.ps1 [reps]
# (default 5). Results verified each rep.

param(
    [int]$Reps = 5,
    [switch]$Quick
)

function Get-Fib {
    param([int]$n)
    if ($n -lt 2) { return $n }
    return (Get-Fib ($n - 1)) + (Get-Fib ($n - 2))
}

function Get-LoopAdd {
    param([int]$iterations)
    $total = 0
    $counter = 1
    for ($i = 0; $i -lt $iterations; $i++) {
        $total += $counter
        $counter += 1
    }
    return $total
}

function Get-StrBuild {
    param([int]$steps)
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append("x")
    for ($i = 0; $i -lt $steps; $i++) {
        [void]$sb.Append("y")
    }
    return $sb.ToString()
}

function Get-ReplaceAll {
    param([string]$text, [string]$old, [string]$new)
    while ($true) {
        $pos = $text.IndexOf($old, [System.StringComparison]::Ordinal)
        if ($pos -eq -1) { return $text }
        $text = $text.Substring(0, $pos) + $new + $text.Substring($pos + $old.Length)
    }
}

function Get-Median {
    param([double[]]$samples)
    $sorted = $samples | Sort-Object
    $n = $sorted.Count
    if ($n % 2 -eq 1) { return $sorted[[int](($n - 1) / 2)] }
    return ($sorted[[int]($n / 2 - 1)] + $sorted[[int]($n / 2)]) / 2.0
}

if ($Quick) {
    $fibN = 16; $loopN = 2000; $strbN = 400; $replN = 15
} else {
    $fibN = 24; $loopN = 20000; $strbN = 2000; $replN = 60
}

# correctness gate
$fibExpected = $(if ($fibN -eq 16) { 987 } else { 46368 })
if ((Get-Fib $fibN) -ne $fibExpected) { throw "fib wrong" }
if ((Get-LoopAdd $loopN) -ne ($loopN * ($loopN + 1) / 2)) { throw "loop wrong" }
if (((Get-StrBuild $strbN) -split "y").Count - 1 -ne $strbN) { throw "strb wrong" }

$hay = ""
for ($i = 0; $i -lt $replN; $i++) { $hay += "cat " }
if (((Get-ReplaceAll $hay "cat " "dog ") -split "d").Count - 1 -ne $replN) { throw "repl wrong" }

$fibSamples = @(); $loopSamples = @(); $strbSamples = @(); $replSamples = @()

for ($rep = 0; $rep -lt $Reps; $rep++) {
    $sw = New-Object System.Diagnostics.Stopwatch

    $sw.Start(); $r1 = Get-Fib $fibN; $sw.Stop()
    $fibSamples += $sw.Elapsed.TotalMilliseconds
    if ($r1 -ne $fibExpected) { throw "fib drifted" }

    $sw.Restart(); $r2 = Get-LoopAdd $loopN; $sw.Stop()
    $loopSamples += $sw.Elapsed.TotalMilliseconds
    if ($r2 -ne ($loopN * ($loopN + 1) / 2)) { throw "loop drifted" }

    $sw.Restart(); $s = Get-StrBuild $strbN; $sw.Stop()
    $r3 = ($s -split "y").Count - 1
    $strbSamples += $sw.Elapsed.TotalMilliseconds
    if ($r3 -ne $strbN) { throw "strb drifted" }

    $hay = ""
    for ($i = 0; $i -lt $replN; $i++) { $hay += "cat " }
    $sw.Restart(); $r4 = ((Get-ReplaceAll $hay "cat " "dog ") -split "d").Count - 1; $sw.Stop()
    $replSamples += $sw.Elapsed.TotalMilliseconds
    if ($r4 -ne $replN) { throw "repl drifted" }
}

$inv = [System.Globalization.CultureInfo]::InvariantCulture
"PowerShell $($PSVersionTable.PSVersion)"
"recursive fib          {0,10:N3} ms" -f ([double](Get-Median $fibSamples)).ToString("F3", $inv)
"loop add               {0,10:N3} ms" -f ([double](Get-Median $loopSamples)).ToString("F3", $inv)
"string build (concat)  {0,10:N3} ms" -f ([double](Get-Median $strbSamples)).ToString("F3", $inv)
"replace all            {0,10:N3} ms" -f ([double](Get-Median $replSamples)).ToString("F3", $inv)
