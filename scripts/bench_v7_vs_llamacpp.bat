@echo off
echo ==================================================
echo v7 inference_example: q8, temp 0, n 100, 30 rounds
echo ==================================================
cd /d %~dp0..\
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is the difference between the airbus a380 and other birds, where does it build its nest, how does it have sex?" -t 0 -n 100 -q8 -r 1

echo ==================================================
echo llama.cpp llama-cli: q8_0 gguf, temp 0, n 100, single turn
echo ==================================================
rem external llama.cpp install: drop llama-cli.exe + the gguf under llamacpp\
llamacpp\llama\llama-cli.exe -m llamacpp\models\Qwen3-0.6B-Q8_0.gguf -p "what is the difference between the airbus a380 and other birds, where does it build its nest, how does it have sex?" --temp 0 -n 100 -ngl 0 -st