// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* test_image_multiturn.c -- 3-turn interleaved vision smoke (chat_image -> generate_image -> chat_image).
 *
 * Exercises the M-RoPE save/restore in generate.inc across the public API:
 * turn 1 v7_chat_image (reset=1, image context established), turn 2
 * v7_generate_image (image-only continuation of the SAME image), turn 3
 * v7_chat_image (reset=0, text follow-up on the shared image context).
 * All three turns must stay ON TOPIC about the image -- no corruption,
 * no repetition degeneration, no control-token soup.
 *
 * usage: test_image_multiturn.exe [-q8 0|1] <dll_path> <db_path> <image.raw>
 *   -q8 0|1     quantize mode (default 0 = F32; both must stay coherent per AGENTS.md 131)
 *   dll_path    path to v7_threading.dll (REQUIRED, no default)
 *   db_path     path to lz4ht.db (REQUIRED, no default)
 *   image_path  path to .raw image (+ matching .meta sidecar, REQUIRED)
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define OUTPUT_BUF_SIZE 8192

/* copy image path with ".raw" -> ".meta" (sidecar), same scheme as dbg_image_recog.c */
void build_meta_path(char *image_path, char *meta_out, int out_sz)
{
	int len = (int)strlen(image_path);
	if (out_sz == 0)
		return;
	if ((len >= 4) && (strcmp(image_path + len - 4, ".raw") == 0))
	{
		len -= 4; /* drop ".raw" */
	}
	if (len + 6 >= out_sz)
	{
		len = out_sz - 6; /* room for ".meta\0" */
	}
	memcpy(meta_out, image_path, len);
	memcpy(meta_out + len, ".meta", 6); /* includes NUL */
}

/* parse a positive integer, skipping leading whitespace, advancing the cursor */
int parse_int(char **cursor)
{
	char *s = *cursor;
	int value = 0;

	while ((*s == ' ') || (*s == '\n') || (*s == '\r') || (*s == '\t'))
	{
		s++;
	}
	while ((*s >= '0') && (*s <= '9'))
	{
		value = value * 10 + (*s - '0');
		s++;
	}
	*cursor = s;
	return value;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	char *dll_path = 0;
	char *db_path = 0;
	char *image_path = 0;
	int q8 = 0;

	int positional = 0;
	for (int i = 1; i < argc; i++)
	{
		if ((strcmp(argv[i], "-q8") == 0) && (i + 1 < argc))
		{
			q8 = atoi(argv[++i]) ? 1 : 0;
			continue;
		}
		switch (positional)
		{
		case 0:
			dll_path = argv[i];
			break;
		case 1:
			db_path = argv[i];
			break;
		case 2:
			image_path = argv[i];
			break;
		}
		positional++;
	}
	if ((!dll_path) || (!db_path) || (!image_path))
	{
		printf("Usage: test_image_multiturn.exe [-q8 0|1] <dll_path> <db_path> <image.raw>\n");
		return 1;
	}

	printf("dll: %s\n", dll_path);
	printf("loading: %s\n", db_path);
	printf("image: %s\n", image_path);
	printf("mode: %s\n", q8 ? "Q8" : "F32");

	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("[FATAL] v7_load failed for %s\n", dll_path);
		return 1;
	}
	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] v7_init died for %s: %s\n", db_path, dll.v7_get_death_reason());
		else
			printf("[FATAL] v7_init failed for %s\n", db_path);
		v7_unload(&dll);
		return 1;
	}

	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real return
	 * AND the death flag must both be checked. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}

	dll.v7_set_quantize(q8);
	dll.v7_set_max_gen(0); /* unlimited: every turn runs until EOS or the output buffer (Sec.154) */

	/* read .meta sidecar for dimensions */
	char meta_path[260];
	build_meta_path(image_path, meta_path, (int)sizeof(meta_path));

	FILE *meta_file = fopen(meta_path, "rb");
	if (!meta_file)
	{
		printf("[FATAL] .meta sidecar open failed: %s\n", meta_path);
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	char meta_content[64];
	int meta_read = (int)fread(meta_content, 1, sizeof(meta_content) - 1, meta_file);
	fclose(meta_file);
	meta_content[meta_read] = 0;
	char *cursor = meta_content;
	int img_width = parse_int(&cursor);
	int img_height = parse_int(&cursor);
	printf("dims: %d x %d\n", img_width, img_height);

	/* read the packed RGB buffer */
	int raw_bytes = img_width * img_height * 3;
	unsigned char *raw_rgb = (unsigned char *)malloc((size_t)raw_bytes);
	if (!raw_rgb)
	{
		printf("[FATAL] malloc failed for pixel buffer\n");
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	FILE *raw_file = fopen(image_path, "rb");
	if (!raw_file)
	{
		printf("[FATAL] .raw image open failed: %s\n", image_path);
		free(raw_rgb);
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	int got = (int)fread(raw_rgb, 1, (size_t)raw_bytes, raw_file);
	fclose(raw_file);
	if (got != raw_bytes)
	{
		printf("[FATAL] .raw short read: %d of %d bytes\n", got, raw_bytes);
		free(raw_rgb);
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}

	char output[OUTPUT_BUF_SIZE];
	int coherent = 1;

	/* === TURN 1: v7_chat_image, reset=1 -- image + question, fresh context === */
	printf("\n... TURN 1: v7_chat_image \"What is in this image?\" (reset=1) ...\n");
	memset(output, 0, sizeof(output));
	dll.v7_chat_image("What is in this image?", 0, raw_rgb, img_width, img_height, output, (int)sizeof(output), 1);
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died in turn 1: %s\n", dll.v7_get_death_reason());
		free(raw_rgb);
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TURN1] pos_after=%d\n", dll.v7_get_position());
	printf("[TURN1] output:\n%s\n\n", output);
	if (!strstr(output, "cat"))
	{
		coherent = 0;
		printf("[FAIL] turn 1 never mentions a cat\n");
	}

	/* === TURN 2: v7_generate_image -- continue the SAME image context, no reset === */
	printf("... TURN 2: v7_generate_image \"Describe the colors of the animal.\" ...\n");
	memset(output, 0, sizeof(output));
	dll.v7_generate_image("Describe the colors of the animal.", raw_rgb, img_width, img_height, output, (int)sizeof(output));
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died in turn 2: %s\n", dll.v7_get_death_reason());
		free(raw_rgb);
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TURN2] pos_after=%d\n", dll.v7_get_position());
	printf("[TURN2] output:\n%s\n\n", output);
	if (!output[0])
	{
		coherent = 0;
		printf("[FAIL] turn 2 produced no output\n");
	}

	/* === TURN 3: v7_chat_image, reset=0 -- text follow-up on the shared context === */
	printf("... TURN 3: v7_chat_image \"How many ears does it have?\" (reset=0) ...\n");
	memset(output, 0, sizeof(output));
	dll.v7_chat_image("How many ears does it have?", 0, raw_rgb, img_width, img_height, output, (int)sizeof(output), 0);
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died in turn 3: %s\n", dll.v7_get_death_reason());
		free(raw_rgb);
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TURN3] pos_after=%d\n", dll.v7_get_position());
	printf("[TURN3] output:\n%s\n\n", output);
	if (!output[0])
	{
		coherent = 0;
		printf("[FAIL] turn 3 produced no output\n");
	}

	free(raw_rgb);
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);

	if (coherent)
	{
		printf("\n[SMOKE] interleaved image multiturn: coherent\n");
		return 0;
	}
	printf("\n[SMOKE] interleaved image multiturn: DEGRADED\n");
	return 1;
}
