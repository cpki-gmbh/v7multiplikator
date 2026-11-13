;;; bench_fib.scm -- SectorLogo multi-language benchmark, GNU Guile edition.
;;; Same four workloads and the same methodology as bench_fib.py: correctness
;;; gate first (doubles as warmup), then N reps with all four workloads timed
;;; per rep, median reported. Every rep re-verifies the expected result.
;;; Usage: guile -s bench_fib.scm [reps] [--quick]

(use-modules (ice-9 regex))
(use-modules (ice-9 format))
(use-modules (srfi srfi-1))

(define (fib n)
  (if (< n 2)
      n
      (+ (fib (- n 1)) (fib (- n 2)))))

(define (loop-add iterations)
  (let loop ((i 0) (total 0) (counter 1))
    (if (= i iterations)
        total
        (loop (+ i 1) (+ total counter) (+ counter 1)))))

;;; one concat per step, exactly like the Python/Tcl/Lisp legs
(define (strbuild steps)
  (let loop ((i 0) (s "x"))
    (if (= i steps)
        s
        (loop (+ i 1) (string-append s "y")))))

;;; replace-all driven to completion, one Search+Replace pair per occurrence
(define (replace-all text old new)
  (let loop ((t text))
    (let ((pos (string-contains t old)))
      (if (not pos)
          t
          (loop (string-append (substring t 0 pos)
                               new
                               (substring t (+ pos (string-length old)))))))))

;;; native single-char count, the same shape as Python's s.count("y") and
;;; Tcl's regexp -all {y} -- so the count is not an interpreted walk here the
;;; way it is in the SectorLogo leg
(define (count-char s ch)
  (string-count s ch))

(define (build-hay n)
  (let loop ((i 0) (hay ""))
    (if (= i n)
        hay
        (loop (+ i 1) (string-append hay "cat ")))))

(define (median samples)
  (let* ((sorted (sort (list-copy samples) <))
         (n (length sorted))
         (mid (quotient n 2)))
    (if (odd? n)
        (list-ref sorted mid)
        (/ (+ (list-ref sorted (- mid 1)) (list-ref sorted mid)) 2.0))))

;;; get-internal-real-time counts in nanoseconds on this platform (calibrated
;;; against a known-duration sleep), so nanoseconds / 1e6 gives milliseconds
(define (elapsed-ms start end)
  (/ (- end start) 1000000.0))

(define (main)
  (let* ((args (cdr (command-line)))   ; drop the script path
         (reps (if (and (pair? args) (string->number (car args)))
                   (inexact->exact (string->number (car args)))
                   5))
         (quick (and (pair? (cdr args))
                     (string=? (cadr args) "--quick")))
         (fib-n (if quick 16 24))
         (loop-n (if quick 2000 20000))
         (strbuild-n (if quick 400 2000))
         (replace-n (if quick 15 60))
         (fib-expected (if quick 987 46368))
         (loop-expected (/ (* loop-n (+ loop-n 1)) 2))
         (fib-samples '())
         (loop-samples '())
         (strb-samples '())
         (repl-samples '()))

    ;; correctness gate (doubles as warmup -- every language's script does this)
    (unless (= (fib fib-n) fib-expected) (error "fib wrong" (fib fib-n)))
    (unless (= (loop-add loop-n) loop-expected) (error "loop add wrong" (loop-add loop-n)))
    (unless (= (count-char (strbuild strbuild-n) #\y) strbuild-n)
      (error "strbuild wrong" strbuild-n))
    (unless (= (count-char (replace-all (build-hay replace-n) "cat " "dog ") #\d) replace-n)
      (error "replace all wrong" replace-n))

    (do ((rep 0 (+ rep 1))) ((= rep reps))
      (let* ((t0 (get-internal-real-time))
             (r1 (fib fib-n))
             (t1 (get-internal-real-time))
             (r2 (loop-add loop-n))
             (t2 (get-internal-real-time))
             (r3 (count-char (strbuild strbuild-n) #\y))
             (t3 (get-internal-real-time))
             (r4 (count-char (replace-all (build-hay replace-n) "cat " "dog ") #\d))
             (t4 (get-internal-real-time)))
        (unless (= r1 fib-expected) (error "fib drifted" r1))
        (unless (= r2 loop-expected) (error "loop add drifted" r2))
        (unless (= r3 strbuild-n) (error "strbuild drifted" r3))
        (unless (= r4 replace-n) (error "replace all drifted" r4))
        (set! fib-samples (cons (elapsed-ms t0 t1) fib-samples))
        (set! loop-samples (cons (elapsed-ms t1 t2) loop-samples))
        (set! strb-samples (cons (elapsed-ms t2 t3) strb-samples))
        (set! repl-samples (cons (elapsed-ms t3 t4) repl-samples))))

    (format #t "GNU Guile ~a~%" (effective-version))
    (format #t "recursive fib          ~10,3f ms~%" (median fib-samples))
    (format #t "loop add               ~10,3f ms~%" (median loop-samples))
    (format #t "string build (concat)  ~10,3f ms~%" (median strb-samples))
    (format #t "replace all            ~10,3f ms~%" (median repl-samples))))

(main)
