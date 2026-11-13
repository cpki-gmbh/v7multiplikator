// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_decide_example.c -- typed decision on a v7 model.
 * Builds the raw decision prompt, runs ONE one-token raw generate (prefill),
 * then reads zero-shot probabilities over the option-label tokens via
 * v7_get_token_probs. No chat template, no sidecar, no training.
 * Usage: v7_decide_example.exe <dll> <db> <kind> <state> <question> "opt1;opt2;..." [-q8 0|1]
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_OPTIONS 16          /* native decision labels A-P */
#define PROMPT_BUF_SIZE 16384
#define OUT_BUF_SIZE 8          /* one token slot: generate must stop after
                                 * sampling the decision token so the RunState
                                 * logits buffer still holds the decision
                                 * position (plan step V2 verifies the stop
                                 * order; if the loop over-forwards, use 1) */

int main(int argc, char **argv)
{
	if (argc < 7)
	{
		printf("Usage: %s <dll_path> <db_path> <kind> <state> <question> \"opt1;opt2;...\" [-q8 0|1]\n", argv[0]);
		printf("kind: noul (2 options) | score (6 options) | choice (2-16 options)\n");
		return 1;
	}

	int use_q8 = 0;
	if (argc >= 9 && strcmp(argv[7], "-q8") == 0)
		use_q8 = atoi(argv[8]);

	char *options[MAX_OPTIONS];
	int num_options = 0;

	char *split = strtok(argv[6], ";");
	while (split && num_options < MAX_OPTIONS)
	{
		options[num_options] = split;
		num_options++;
		split = strtok(0, ";");
	}
	if (num_options < 2)
	{
		printf("need at least 2 options (semicolon-separated)\n");
		return 1;
	}

	printf("Step 1: load DLL\n");
	struct v7_lib dll;
	if (v7_load(argv[1], &dll) != 0)
	{
		printf("load failed\n");
		return 1;
	}

	printf("Step 2: V7_init\n");
	if (!dll.v7_init(argv[2]))
	{
		if (dll.v7_engine_is_dead())
			printf("V7_init died: %s\n", dll.v7_get_death_reason());
		else
			printf("V7_init failed\n");
		v7_unload(&dll);
		return 1;
	}

	printf("Step 3: initialize inference\n");
	if (!dll.v7_initialize_inference())
	{
		if (dll.v7_engine_is_dead())
			printf("inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}
	dll.v7_set_quantize(use_q8);

	/* verbalizer ids: every option label must encode to exactly one token */
	int label_ids[MAX_OPTIONS];
	for (int i = 0; i < num_options; i++)
	{
		int label_tokens[8];
		int num_label_tokens = 0;
		dll.v7_encode(options[i], label_tokens, &num_label_tokens, 8);
		if (num_label_tokens != 1)
		{
			printf("option '%s' encodes to %d tokens -- labels must be single-token\n", options[i], num_label_tokens);
			v7_unload(&dll);
			return 1;
		}
		label_ids[i] = label_tokens[0];
	}

	/* raw decision prompt -- no chat template (typed decision-prompt shape).
	 * snprintf + a clamp: the kind/state/question come straight from the
	 * command line, and an over-long argument must truncate, never smash
	 * this 16 KB stack buffer. */
	char prompt[PROMPT_BUF_SIZE];
	int plen = snprintf(prompt, sizeof(prompt), "[kind] %s\n[state] %s\n[question] %s\n[options]\n", argv[3], argv[4], argv[5]);
	if (plen < 0)
		plen = 0;
	if (plen > (int)sizeof(prompt) - 1)
		plen = (int)sizeof(prompt) - 1;
	for (int i = 0; i < num_options && plen < (int)sizeof(prompt) - 256; i++)
	{
		int w = snprintf(prompt + plen, sizeof(prompt) - plen, "%c) %s\n", 'A' + i, options[i]);
		if (w < 0 || w >= (int)sizeof(prompt) - plen)
			break; /* did not fit -- stop listing, the prompt stays bounded */
		plen += w;
	}
	if (plen > (int)sizeof(prompt) - 16)
		plen = (int)sizeof(prompt) - 16;
	snprintf(prompt + plen, sizeof(prompt) - plen, "\n[decision]:");

	/* fresh KV: the logits buffer then holds the raw decision-position
	 * logits when generate stops after one token slot. sampler untouched:
	 * the engine defaults (temperature 0.0 greedy, penalty 0) are what this
	 * decision path needs (AGENTS.md 6a rule 12). */
	dll.v7_set_position(0);

	char output[OUT_BUF_SIZE];
	dll.v7_generate(prompt, output, OUT_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died during generate: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}

	/* zero-filled: a get_token_probs that skips an option must not argmax
	 * over stack garbage (same rationale as the /v1/decide server handler) */
	float probs[MAX_OPTIONS] = { 0 };
	dll.v7_get_token_probs(label_ids, num_options, probs);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died during get_token_probs: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}

	int best_index = 0;
	for (int i = 1; i < num_options; i++)
	{
		if (probs[i] > probs[best_index])
			best_index = i;
	}

	printf("{\"options\":[");
	for (int i = 0; i < num_options; i++)
		printf("%s\"%s\"", i ? "," : "", options[i]);
	printf("],\"probabilities\":[");
	for (int i = 0; i < num_options; i++)
		printf("%s%.4f", i ? "," : "", probs[i]);
	printf("],\"choice\":\"%s\"}\n", options[best_index]);

	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
