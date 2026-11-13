@echo off
rem test_bayes_gate.bat -- classify the probe questions with the naive bayes
rem gate: class0 = ardmark corpus (inject side), class1 = everyday world
rem corpus (suppress side). Diagnostics with the score margins land in
rem bin\bayes_classify.log. Run from the project ROOT.

echo === off-topic probes (want class1) ===
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "By the way, what is the capital of France and when was it founded?"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "what is the capital of France"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "what metal is best for tools and why"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "How long do I boil pasta?"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "Tell me why cats purr."
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "What is the capital of Italy and who founded Rome?"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "Who won the last football world cup?"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "How do I cook rice so it is not mushy?"

echo === on-topic probes (want class0) ===
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "Are there dragons in Ardmark? Tell me of them."
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "Tell me about the winters in Ardmark. Are they hard on your people?"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "who ruled ardmark after queen maren and what did her son do"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "what do the miners fear down in the copper mines"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "who built the amber palace and when"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "what is the language of ardmark called"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "tell me about the civil war, why did it start and how did it end"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "What happens at the Feast of Lanterns?"
bin\bayes_classify.exe classify examples\ardmark\ardmark_bayes_state.txt "Tell me of the herring fleets and the harbour."
