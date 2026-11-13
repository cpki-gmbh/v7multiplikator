# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

# followup.py -- a CONTINUATION of the running conversation with the v7
# server (same session, thinking ON). React naturally to what the model said
# before: it talked about a person, ask about that person; a dragon, follow
# the story.
# Usage: python followup.py "your next message"
# Prints the COMPLETE raw JSON response (thinking + visible answer + tokens
# + anything else the server returns) straight to stdout -- nothing filtered,
# nothing shortened (Sec.195 rule 5).

import json
import sys
import urllib.request

if len(sys.argv) < 2:
    print("Usage: python followup.py \"next message\"")
    sys.exit(1)

payload = {
    "messages": [{"role": "user", "content": sys.argv[1]}],
    "max_tokens": 0,   # 0 = the server's unlimited default
    "reset": 0,        # continue: the running session stays
    "thinking": 1,
}

request = urllib.request.Request(
    "http://localhost:8080/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=3600) as response:
    body = response.read()

data = json.loads(body.decode("utf-8", errors="replace"))


def readable(text):
    """Unescape line breaks in every form the server emits: the JSON
    decodes \r\n pairs to real breaks, the model's own text carries
    raw backslash-n sequences that print as '\n' text."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    return text


for choice in data.get("choices", []):
    message = choice.get("message", {})
    reasoning = message.get("reasoning_content", "")
    answer = message.get("content", "")

    print("")
    print("================= THINKING =================")
    print(readable(reasoning))
    print("================= ANSWER ==================")
    print(readable(answer))
    print("================= END =====================")

usage = data.get("usage", {})
print("tokens: prompt=%s completion=%s total=%s" % (
    usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")))
