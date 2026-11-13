tools\genome_cycle_sort.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --shape 1024x3072 --rounds 100 --mutate-percentage 3
tools\genome_cycle_sort.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --shape 1024x3072 --rounds 200 --mutate-percentage 3
tools\genome_cycle_sort.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --shape 1024x3072 --rounds 500 --mutate-percentage 3 --sorter-stdout-file build_logs\cycle_sort_500.txt
