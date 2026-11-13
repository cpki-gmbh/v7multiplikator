# Typed Decisions -- Measured Results

Everything below was produced by the shipped binaries
(`bin\v7_decide_example.exe`, `bin\spam_classifier.exe`) running
Qwen3-0.6B-Instruct, greedy, penalty off, fresh position per decision.
Every row is reproducible with one command; greedy decisions give the same
numbers every time (checked, see section 4).

## 1. What a decision run actually does

The prompt names a kind, a state, one question and the options. The engine
prefills it like any other prompt. At that moment the logits for the next
token sit in the state buffer, and the option labels are tokens like any
other. We read the logits for those tokens and normalize them. That is the
whole trick: no generation beyond the single decision token, no chat
template, no training.

## 2. Yes/no -- the "sky is blue" family

| state / question | P(false) | P(true) | decision | correct? |
|---|---|---|---|---|
| "The sky is blue." / "Is the sky blue?" | 0.0158 | 0.9842 | true | yes |
| same, 8-bit forward | 0.0176 | 0.9824 | true | yes |
| "The sky is green." / "Is the sky blue?" | 0.0660 | 0.9340 | true | NO |
| "The sky is green." / "Is that statement true?" | 0.0595 | 0.9405 | true | NO |

### 2.1 Why the green sky still reads "true"

We expected the model to notice that a green sky contradicts the question.
It does not. It reads the question and completes it: people ask "is the sky
blue?" because the answer is yes, so the next token is "true" no matter what
the state says. Rephrasing to "Is that statement true?" changed nothing
(0.94 for true). At 0.6B parameters the model simply does not check a claim
against supplied context. Published work on decision prompts saw the same
weakness in comparable models before anyone trained on the task.

The practical takeaway: don't use yes/no decisions to verify facts with a
stock model. Yes/no works when the question has a natural yes-answer and the
state supplies something the model can actually weigh. Fact-checking needs
the trained variant (planned as database data, no engine change).

## 3. Choosing between options

| test | probabilities | choice | correct? |
|---|---|---|---|
| "Customer: my card was charged twice for one coffee." / "Which team should handle this?" / billing, shipping, support | 0.6312, 0.3078, 0.0610 | billing | yes |

This is the case that works well out of the box. The right option wins
clearly, and the three probabilities add up to 1.0 as they should.

## 4. Same numbers every time

Running the same command twice gives the same numbers to the last digit, in
both forward modes. We also ran a decision, then a two-turn chat, then the
same decision again: 0.6261 / 0.3001 / 0.0738 before the chat, exactly the
same after. Each decision resets its own position, so earlier conversation
cannot bleed into it.

## 5. Full precision vs 8-bit

The sky-blue prompt reads 0.0158/0.9842 in full precision and
0.0176/0.9824 in 8-bit. That is a shift of about two tenths of a percent on
the margin -- small, but it is there. If you operate on a threshold, pick it
in the mode you ship.

## 6. Sorting mail: spam or not

Six mails, three spam and three normal, one decision pass each:

| mail | P(ham) | P(spam) | decision | correct? |
|---|---|---|---|---|
| spam1_lottery.txt (lottery win, "act NOW") | 0.1558 | 0.8442 | spam | yes |
| spam2_phishing.txt (account suspension, phishing link) | 0.1021 | 0.8979 | spam | yes |
| spam3_pills.txt (pharmacy spam, ALL-CAPS offer) | 0.3651 | 0.6349 | spam | yes |
| ham1_meeting_notes.txt (sprint review notes) | 0.9979 | 0.0021 | ham | yes |
| ham2_appointment.txt (clinic appointment letter) | 0.9550 | 0.0450 | ham | yes |
| ham3_friends.txt (private hiking-trip mail) | 0.9999 | 0.0001 | ham | yes |

### 6.1 What the numbers say

All six came out right. The distances are worth a look, though.

Normal mail is nowhere near the spam side: the three ham mails sit between
0.955 and 0.9999. Ordinary prose is easy for the model to tell apart.

The two obvious scams land at 0.84 and 0.90. You would not want a lower
threshold than that if you also care about the ham side, but nothing
legitimate in this set came anywhere close.

The pharmacy mail only reached 0.63. Words like "offer" and "shipping" show
up in real marketing mail too, so the model hesitates; the ALL-CAPS text and
the pill-shop URL are what push it over. If your threshold sat above 0.65
you would deliver this one to the inbox. That is the honest picture: the
zero-shot numbers are a starting point, and a real deployment tunes its
cutoff on mail it actually receives.

## 7. What we learned about writing the prompts

1. Labels have to be single tokens. "tech support" becomes two tokens and
   the tool refuses it; "support" alone works.
2. One question with several options beats several yes/no questions. The
   3-way routing test resolves in one pass.
3. Yes/no cannot fact-check on a stock model (section 2.1). The question
   sounds answered before the state is read.
4. Order and wording move a few percent. Comparable published work reports
   about 7% of answers flipping when options are shuffled; for a two-way
   call, ask both ways and average.
5. Put the content in the state and keep the question short. The whole mail
   goes into the state; the question stays one line.
