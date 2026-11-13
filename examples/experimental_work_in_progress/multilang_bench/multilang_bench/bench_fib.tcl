# Tcl 9.0 cross-language benchmark -- same 4 workloads, warmup + N reps, median.
# Usage: tclsh bench_fib.tcl [reps] (default 5). Results verified each rep.

proc fib {n} {
    if {$n < 2} { return $n }
    return [expr {[fib [expr {$n - 1}]] + [fib [expr {$n - 2}]]}]
}

proc loop_add {iterations} {
    set total 0
    set counter 1
    for {set i 0} {$i < $iterations} {incr i} {
        set total [expr {$total + $counter}]
        incr counter
    }
    return $total
}

proc strbuild {steps} {
    set s "x"
    for {set i 0} {$i < $steps} {incr i} {
        append s "y"
    }
    return $s
}

proc replace_all {text old new} {
    while {1} {
        set pos [string first $old $text]
        if {$pos == -1} { return $text }
        set text [string replace $text $pos [expr {$pos + [string length $old] - 1}] $new]
    }
}

proc median {samples} {
    set sorted [lsort -real $samples]
    set n [llength $sorted]
    if {$n % 2 == 1} { return [lindex $sorted [expr {$n / 2}]] }
    return [expr {([lindex $sorted [expr {$n / 2 - 1}]] + [lindex $sorted [expr {$n / 2}]]) / 2.0}]
}

proc now_ms {} {
    return [expr {[clock clicks -microseconds] / 1000.0}]
}

set reps 5
if {$argc >= 1} { set reps [lindex $argv 0] }
set quick [expr {$argc >= 2 && [lindex $argv 1] eq "--quick"}]

if {$quick} {
    set fib_n 16; set loop_n 2000; set strb_n 400; set repl_n 15
} else {
    set fib_n 24; set loop_n 20000; set strb_n 2000; set repl_n 60
}

# correctness gate
set fib_expected [expr {$fib_n == 16 ? 987 : 46368}]
if {[fib $fib_n] != $fib_expected} { puts "fib wrong"; exit 1 }
if {[loop_add $loop_n] != [expr {$loop_n * ($loop_n + 1) / 2}]} { puts "loop wrong"; exit 1 }
set built [strbuild $strb_n]
if {[regexp -all {y} $built] != $strb_n} { puts "strb wrong"; exit 1 }
set hay ""
for {set i 0} {$i < $repl_n} {incr i} { append hay "cat " }
if {[regexp -all {d} [replace_all $hay "cat " "dog "]] != $repl_n} { puts "repl wrong"; exit 1 }

set fib_samples {}; set loop_samples {}; set strb_samples {}; set repl_samples {}

for {set rep 0} {$rep < $reps} {incr rep} {
    set t0 [now_ms]
    set r1 [fib $fib_n]
    set t1 [now_ms]
    set r2 [loop_add $loop_n]
    set t2 [now_ms]
    set s [strbuild $strb_n]
    set r3 [regexp -all {y} $s]
    set t3 [now_ms]
    set hay ""
    for {set i 0} {$i < $repl_n} {incr i} { append hay "cat " }
    set r4 [regexp -all {d} [replace_all $hay "cat " "dog "]]
    set t4 [now_ms]

    if {$r1 != $fib_expected} { puts "fib drifted"; exit 1 }
    if {$r2 != [expr {$loop_n * ($loop_n + 1) / 2}]} { puts "loop drifted"; exit 1 }
    if {$r3 != $strb_n} { puts "strb drifted"; exit 1 }
    if {$r4 != $repl_n} { puts "repl drifted"; exit 1 }

    lappend fib_samples [expr {$t1 - $t0}]
    lappend loop_samples [expr {$t2 - $t1}]
    lappend strb_samples [expr {$t3 - $t2}]
    lappend repl_samples [expr {$t4 - $t3}]
}

puts "Tcl [info patchlevel]"
puts [format "recursive fib          %10.3f ms" [median $fib_samples]]
puts [format "loop add               %10.3f ms" [median $loop_samples]]
puts [format "string build (concat)  %10.3f ms" [median $strb_samples]]
puts [format "replace all            %10.3f ms" [median $repl_samples]]
