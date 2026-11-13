set v7_override_prio=0

bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is the capital of France and why is it important in European history?" -t 0 -n 100 -q8 -r 3
