// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* spam_classifier.c -- classify an email text file as spam or not spam
 * using a v7 typed decision (zero-shot, one prefill pass).
 * Reads the mail text, builds a decision prompt with the mail as the state,
 * and prints the P(spam) / P(ham) probabilities and the decision.
 * Usage: spam_classifier.exe <dll_path> <db_path> <mail.txt> [-q8 0|1]
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAIL_BUF_SIZE 32768
#define PROMPT_BUF_SIZE 65536
#define OUT_BUF_SIZE 8          /* one token slot: generate stops right after
                                 * sampling the decision token, so the RunState
                                 * logits buffer still holds the decision
                                 * position when v7_get_token_probs reads it */

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		printf("Usage: %s <dll_path> <db_path> <mail.txt> [-q8 0|1]\n", argv[0]);
		return 1;
	}

	int use_q8 = 0;
	if (argc >= 6 && strcmp(argv[4], "-q8") == 0)
		use_q8 = atoi(argv[5]);

	FILE *mail_file = fopen(argv[3], "rb");
	if (!mail_file)
	{
		printf("cannot open %s\n", argv[3]);
		return 1;
	}

	static char mail_text[MAIL_BUF_SIZE];
	int mail_len = (int)fread(mail_text, 1, MAIL_BUF_SIZE - 1, mail_file);
	fclose(mail_file);
	mail_text[mail_len] = '\0';

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

	/* verbalizer ids: ham / spam must each be a single token */
	char *option_labels[2] = { "ham", "spam" };
	int label_ids[2];
	for (int i = 0; i < 2; i++)
	{
		int label_tokens[8];
		int num_label_tokens = 0;
		dll.v7_encode(option_labels[i], label_tokens, &num_label_tokens, 8);
		if (num_label_tokens != 1)
		{
			printf("option '%s' encodes to %d tokens -- labels must be single-token\n", option_labels[i], num_label_tokens);
			v7_unload(&dll);
			return 1;
		}
		label_ids[i] = label_tokens[0];
	}

	/* decision prompt: the mail is the state, the kind is the question */
	static char prompt[PROMPT_BUF_SIZE];
	sprintf(prompt, "[kind] choice\n[state] %s\n[question] Which kind of email is this?\n[options]\nA) ham\nB) spam\n\n[decision]:", mail_text);

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
	float probs[2] = { 0 };
	dll.v7_get_token_probs(label_ids, 2, probs);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died during get_token_probs: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}

	int best_index = (probs[1] > probs[0]) ? 1 : 0;
	printf("{\"file\":\"%s\",\"p_ham\":%.4f,\"p_spam\":%.4f,\"decision\":\"%s\"}\n",
	       argv[3], probs[0], probs[1], option_labels[best_index]);

	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
