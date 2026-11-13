// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * test_multiturn_manual.c -- §150 isolation test
 *
 * turn 1: v7s_chat (reset=1) as usual.
 * turn 2: MANUAL pipeline -- render template in C, encode via DLL export,
 *         per-token forward_f32 calls, greedy sample via v7s_sample.
 *         if the manual path answers about dogs -> bug is in prefill_*.
 *         if manual path also breaks -> bug is in forward_* at pos>250.
 *
 * struct offsets mirror chat.inc (tokenizer: vocab=0,eos=28; sampler:
 * vocab=0,probindex=8,rng=32).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#define MAX_OUT 12000

/* tokenizer field offsets (match chat.inc CHAT_TOK_*) */
#define TK_VOCAB 0
#define TK_EOS_ID 28
/* sampler field offsets (match chat.inc CHAT_SAM_*) */
#define SM_VOCAB 0
#define SM_PROBINDEX 8
#define SM_RNG 32

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Usage: %s <v7_threading.dll> <lz4ht.db>\n", argv[0]);
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = argv[2];

	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
		return 1;
	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] V7_init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] V7_init failed\n");
		v7_unload(&dll);
		return 1;
	}

	/* initialize inference (context hidden in BSS) */
	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real return
	 * AND the death flag must both be checked. A dead engine must never reach
	 * v7_get_internal_inference_ctx() below -- that is a raw context walk. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}
	inference_context *ctx = dll.v7_get_internal_inference_ctx(); /* deep tool: raw struct walking */
	dll.v7_set_quantize(0); /* F32 -- match the traced path */
	dll.v7_set_max_gen(600);
	dll.v7_enable_thinking(1);
	printf("[test] F32, thinking ON\n");

	char output[MAX_OUT];

	/* === TURN 1: normal v7s_chat (reset=1) === */
	printf("... TURN 1 (v7s_chat, reset=1) ...\n");
	memset(output, 0, sizeof(output));
	dll.v7_chat("what is a cat?", 0, output, (int)sizeof(output), 1);
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died in turn 1: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TURN1] pos_after=%d\n", dll.v7_get_position());
	printf("[TURN1] output (%u chars):\n%s\n\n", (unsigned)strlen(output), output);

	/* === TURN 2: MANUAL per-token forward pipeline === */
	printf("... TURN 2 (manual forward_f32) ...\n");
	void *tr = ctx->transformer;
	unsigned char *tok = (unsigned char *)ctx->tokenizer;
	unsigned char *sam = (unsigned char *)ctx->sampler;

	char ***vocab = (char ***)(tok + TK_VOCAB);
	int eos_id = *(int *)(tok + TK_EOS_ID);
	int vsam = *(int *)(sam + SM_VOCAB);
	v7_probindex *pi = *(v7_probindex **)(sam + SM_PROBINDEX);
	unsigned long long *rng = (unsigned long long *)(sam + SM_RNG);

	int pos = dll.v7_get_position();
	printf("[TURN2] starting at pos=%d (eos=%d vocab=%d)\n", pos, eos_id, vsam);

	/* render the multi@think template manually */
	char rendered[2048];
	snprintf(rendered, sizeof(rendered), "<|im_end|>\n<|im_start|>user\nwhat is a dog?<|im_end|>\n<|im_start|>assistant\n");

	/* encode via DLL export */
	int tokens[512];
	int n_tokens = 0;
	dll.v7_encode(rendered, tokens, &n_tokens, 512);
	printf("[TURN2] encoded %d tokens\n", n_tokens);
	for (int i = 0; i < n_tokens; i++)
		printf("[TURN2] tok[%d] pos=%d id=%d piece=\"%s\"\n", i, pos + i, tokens[i], (*vocab)[tokens[i]]);

	/* per-token forward over the prompt */
	float *logits = 0;
	for (int i = 0; i < n_tokens; i++)
	{
		logits = dll.forward_f32(tr, tokens[i], pos + i);
	}
	pos += n_tokens;

	/* generate greedily */
	memset(output, 0, sizeof(output));
	int out_len = 0;
	for (int g = 0; g < 400; g++)
	{
		/* logits probe -- same anchors as the ASM TRACE (endoftext / okay) */
		if (g < 8)
			printf("manual_sample pos=%d logits[eot]=%f logits[okay]=%f\n", pos, logits[151643], logits[32313]);
		int next = dll.v7s_sample(logits, vsam, 0.0f, 0.0f, rng, pi);
		if (g < 8)
			printf("manual_sample tok=%d (%s)\n", next, (*vocab)[next]);
		if (next == eos_id || next == 151643)
			break; /* stop on im_end or endoftext */
		char *piece = (*vocab)[next];
		int plen = strlen(piece);
		if (out_len + plen + 1 >= sizeof(output))
			break;
		memcpy(output + out_len, piece, plen);
		out_len += plen;
		logits = dll.forward_f32(tr, next, pos);
		pos++;
	}
	/* sync the context position after the MANUAL pipeline. the public
	 * v7_set_position is rewind-only by contract (the engine cannot verify
	 * that forward-wound KV slots were written); this differential tool
	 * wrote them itself via the exported forward_f32 calls above, so the
	 * raw ctx->pos write is the sanctioned deep-tool path (Sec.12b). */
	ctx->pos = pos;
	printf("[TURN2] pos_after=%d\n", dll.v7_get_position());
	printf("[TURN2] MANUAL output (%u chars):\n%s\n\n", out_len, output);

	/* v7_shutdown joins the workers BEFORE the heap teardown + FreeLibrary */
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
