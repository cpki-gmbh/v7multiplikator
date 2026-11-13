;;; bench_fib.lsp -- SectorLogo multi-language benchmark, Common Lisp edition.
;;; Same four workloads and the same methodology as bench_fib.py:
;;; correctness gate first (acts as warmup), then N reps with all four
;;; workloads timed per rep, median reported.
;;; Usage: sbcl --script bench_fib.lsp [reps] [--quick]

(declaim (optimize (speed 3) (safety 0) (debug 0) (space 0)))

(defun fib (n)
  (if (< n 2)
    n
    (+ (fib (- n 1)) (fib (- n 2)))))

(defun loop-add (iterations)
  (let ((total 0) (counter 1))
    (dotimes (i iterations total)
      (incf total counter)
      (incf counter))))

(defun strbuild (steps)
  (let ((s "x"))
    (dotimes (i steps s)
      (setf s (concatenate 'string s "y")))))

(defun replace-all (text old new)
  (let ((pos (search old text :test #'char=)))
    (if (null pos)
      text
      (replace-all (concatenate 'string
                                (subseq text 0 pos)
                                new
                                (subseq text (+ pos (length old))))
                   old new))))

(defun median (samples)
  (let* ((sorted (sort (copy-list samples) #'<))
         (n (length sorted))
         (mid (floor n 2)))
    (if (oddp n)
      (nth mid sorted)
      (/ (+ (nth (- mid 1) sorted) (nth mid sorted)) 2.0d0))))

(defun elapsed-ms (start end)
  (/ (* (- end start) 1000.0d0) internal-time-units-per-second))

(defun build-hay (n)
  (let ((hay ""))
    (dotimes (i n hay)
      (setf hay (concatenate 'string hay "cat ")))))

(defun main ()
  (let* ((args (cdr (cdr (cdr sb-ext:*posix-argv*))))  ; skip "sbcl" "--script" "file"
         (reps (if args (parse-integer (first args) :junk-allowed t) 5))
         (quick (and (second args) (string= (second args) "--quick")))
         (fib-n (if quick 16 24))
         (loop-n (if quick 2000 20000))
         (strbuild-n (if quick 400 2000))
         (replace-n (if quick 15 60))
         (fib-samples nil) (loop-samples nil) (strb-samples nil) (repl-samples nil))
    (unless reps (setf reps 5))

    ;; correctness gate (doubles as warmup -- every language's script does this)
    (assert (= (fib fib-n) (if quick 987 46368)))
    (assert (= (loop-add loop-n) (/ (* loop-n (+ loop-n 1)) 2)))
    (assert (= (count #\y (strbuild strbuild-n)) strbuild-n))
    (assert (= (count #\d (replace-all (build-hay replace-n) "cat " "dog ")) replace-n))

    (dotimes (rep reps)
      (let ((t0 (get-internal-real-time))
            (r1 (fib fib-n))
            (t1 (get-internal-real-time))
            (r2 (loop-add loop-n))
            (t2 (get-internal-real-time))
            (r3 (count #\y (strbuild strbuild-n)))
            (t3 (get-internal-real-time))
            (r4 (count #\d (replace-all (build-hay replace-n) "cat " "dog ")))
            (t4 (get-internal-real-time)))
        (declare (ignore r1 r2 r3 r4))
        (push (elapsed-ms t0 t1) fib-samples)
        (push (elapsed-ms t1 t2) loop-samples)
        (push (elapsed-ms t2 t3) strb-samples)
        (push (elapsed-ms t3 t4) repl-samples)))

    (format t "recursive fib          ~10,3f ms~%" (median fib-samples))
    (format t "loop add               ~10,3f ms~%" (median loop-samples))
    (format t "string build (concat)  ~10,3f ms~%" (median strb-samples))
    (format t "replace all            ~10,3f ms~%" (median repl-samples))))

(main)
