// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* dbg_image_recog.c -- standalone C harness: image + text prompt -> caption.
 *
 * reimplementation of examples/dbg_image_recog.asm in C.
 * loads v7_threading.dll -> v7_init -> v7_initialize_inference ->
 * v7_chat_image -> v7_free_inference. the HARNESS loads the .raw
 * image file + .meta sidecar and passes the packed RGB buffer + dimensions
 * to the DLL. the DLL does NO file I/O.
 *
 * usage: dbg_image_recog.exe [options] <dll_path> <db_path> [image_path] [prompt]
 *   -q8 0|1    -- quantize mode: 0=F32, 1=Q8 (default: 0)
 *   -n <tok>   -- generation token budget (default: 0 = unlimited, run until EOS)
 *   dll_path   -- path to v7_threading.dll (REQUIRED, no default)
 *   db_path    -- path to lz4ht.db (REQUIRED, no default)
 *   image_path -- path to .raw image (+ matching .meta sidecar) (default: "cat.raw")
 *   prompt     -- text prompt (default: "What is in this image?")
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define OUTPUT_BUF_SIZE 32768

/* copy image path with ".raw" -> ".meta" (sidecar). mirrors the ASM harness:
 * copies the prefix before ".raw" then appends ".meta". if the path does not
 * end in ".raw", the extension is appended instead. */
void build_meta_path(char *image_path, char *meta_out, int out_sz)
{
	int len = strlen(image_path);
	if (out_sz == 0)
		return;
	if (len >= 4 && strcmp(image_path + len - 4, ".raw") == 0)
	{
		len -= 4; /* drop ".raw" */
	}
	if (len + 6 >= out_sz)
		len = out_sz - 6; /* room for ".meta\0" */
	memcpy(meta_out, image_path, len);
	memcpy(meta_out + len, ".meta", 6); /* includes NUL */
}

/* parse a positive integer, skipping leading space/newline/CR, advancing the
 * cursor past the digits (mirrors dbg_parse_int in the ASM harness). */
int parse_int(char **p)
{
	char *s = *p;
	int value = 0;
	for (;;)
	{
		if (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t')
		{
			s++;
			continue;
		}
		break;
	}
	while (*s >= '0' && *s <= '9')
	{
		value = value * 10 + (*s - '0');
		s++;
	}
	*p = s;
	return value;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	/* phase 1b: parse command line arguments.
	 * flags (-q8 0|1) may appear anywhere; the remaining positional args are
	 * dll_path, db_path, image_path, prompt in order.
	 * dll_path and db_path are REQUIRED (no defaults, per AGENTS.md 106). */
	char *dll_path = 0;
	char *db_path = 0;
	char *image_path = "cat.raw";
	char *prompt = "What is in this image?";
	char *set_template = 0; /* -set-template "<fmt>": override __sys_template (\n decoded) */
	char *set_visfmt = 0;   /* -set-visfmt "prefix|marker|suffix": override __sys_vision_prompt_fmt */
	int q8 = 0;             /* default F32 path; -q8 1 selects Q8 (AGENTS.md 131: both must work) */
	int max_gen = 0;        /* -n value; 0 = unlimited: run until EOS (Sec.154) */

	int positional = 0;
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-q8") == 0 && i + 1 < argc)
		{
			q8 = atoi(argv[++i]) ? 1 : 0;
			continue;
		}
		if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
		{
			max_gen = atoi(argv[++i]);
			continue;
		}
		if (strcmp(argv[i], "-set-template") == 0 && i + 1 < argc)
		{
			set_template = argv[++i];
			continue;
		}
		if (strcmp(argv[i], "-set-visfmt") == 0 && i + 1 < argc)
		{
			set_visfmt = argv[++i];
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
		case 3:
			prompt = argv[i];
			break;
		}
		positional++;
	}
	if (!dll_path || !db_path)
	{
		printf("Usage: dbg_image_recog.exe [-q8 0|1] [-n tokens] <dll_path> <db_path> [image_path] [prompt]\n");
		printf("  dll_path -- path to v7_threading.dll (REQUIRED, no default)\n");
		printf("  db_path  -- path to lz4ht.db (REQUIRED, no default)\n");
		return 1;
	}

	printf("dll: %s\n", dll_path);
	printf("loading: %s\n", db_path);
	printf("question: %s\n", prompt);
	printf("mode: %s\n", q8 ? "Q8" : "F32");

	/* phase 2: load v7_threading.dll + resolve exports. */
	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("[FATAL] LoadLibraryA failed for %s\n", dll_path);
		return 1;
	}

	/* phase 3.5: initialize database via V7_init. */
	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] V7_init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] V7_init failed\n");
		v7_unload(&dll);
		return 1;
	}

	/* test hooks: override DB keys in-memory BEFORE v7s_inference_create copies
	 * them into engine state. avoids multi-GB DB rebuilds while iterating on
	 * prompt formats. -set-template decodes "\n" to a real newline;
	 * -set-visfmt takes "prefix|marker|suffix" -> 3 NUL-terminated strings. */
	if (set_template)
	{
		char buf[1024];
		int o = 0;
		for (char *p = set_template; *p && o + 1 < sizeof(buf); p++)
		{
			if (p[0] == '\\' && p[1] == 'n')
			{
				buf[o++] = '\n';
				p++;
			}
			else
				buf[o++] = *p;
		}
		buf[o] = 0;
		if (!dll.v7_db_add("__sys_template", buf, o + 1, 0))
		{
			printf("[FATAL] v7_db_add(__sys_template) failed\n");
			v7_unload(&dll);
			return 1;
		}
		printf("template overridden: %s\n", buf);
	}
	if (set_visfmt)
	{
		char blob[256];
		int o = 0;
		for (char *p = set_visfmt; *p && o + 1 < sizeof(blob); p++)
		{
			blob[o++] = (*p == '|') ? '\0' : *p; /* '|' separates the 3 strings */
		}
		blob[o++] = '\0'; /* final string terminator */
		if (!dll.v7_db_add("__sys_vision_prompt_fmt", blob, o, 0))
		{
			printf("[FATAL] v7_db_add(__sys_vision_prompt_fmt) failed\n");
			v7_unload(&dll);
			return 1;
		}
		printf("visfmt overridden (%u bytes)\n", (unsigned)o);
	}

	/* phase 4: create inference context -- START TIMING HERE. */
	LARGE_INTEGER freq, t0, t1;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&t0);

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

	/* sampler untouched: the engine defaults are temperature 0.0 (greedy
	 * argmax), topp 0.0, seed 0 -- hardcoded sampler writes are forbidden
	 * (AGENTS.md 6a rule 12). */
	dll.v7_set_quantize(q8); /* 0=F32, 1=Q8 -- from -q8 flag (AGENTS.md 131) */
	dll.v7_set_max_gen(max_gen); /* -n flag; 0 = unlimited: run until EOS (Sec.154) */

	/* phase 5: load .raw image + .meta sidecar. */
	char meta_path[260];
	build_meta_path(image_path, meta_path, sizeof(meta_path));

	/* --- phase 5b: open + parse .meta sidecar ("W H") --- */
	FILE *fmeta = fopen(meta_path, "rb");
	if (!fmeta)
	{
		printf("[FATAL] .meta sidecar open failed\n");
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	char meta_content[64];
	int meta_n = fread(meta_content, 1, sizeof(meta_content) - 1, fmeta);
	fclose(fmeta);
	meta_content[meta_n] = 0;
	char *mc = meta_content;
	int img_width = parse_int(&mc);
	int img_height = parse_int(&mc);

	/* --- phase 5c: read .raw image file --- */
	int raw_bytes = (int)img_width * (int)img_height * 3u;
	unsigned char *raw_rgb = (unsigned char *)malloc(raw_bytes);
	if (!raw_rgb)
	{
		printf("[FATAL] malloc failed for raw pixel buffer\n");
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}

	FILE *fraw = fopen(image_path, "rb");
	if (!fraw)
	{
		printf("[FATAL] .raw image open failed\n");
		free(raw_rgb);
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	int got = fread(raw_rgb, 1, raw_bytes, fraw);
	fclose(fraw);
	if (got != raw_bytes)
	{
		printf("[FATAL] .raw image short read: %u of %u bytes\n", got, raw_bytes);
		free(raw_rgb);
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}

	/* phase 5d/5e: call v7s_chat_image_mem (9 args, no system prompt, reset=1). */
	char output[OUTPUT_BUF_SIZE];
	memset(output, 0, sizeof(output));
	printf("Dbg: calling v7s_chat_image_mem\n");
	dll.v7_chat_image(prompt, 0, raw_rgb, img_width, img_height, output, OUTPUT_BUF_SIZE, 1);
	printf("Dbg: v7s_chat_image_mem returned\n");

	/* END TIMING -- write elapsed seconds to a temp file + stderr. */
	QueryPerformanceCounter(&t1);
	double elapsed = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
	{
		FILE *tf = fopen("bench_time.txt", "w");
		if (tf)
		{
			fprintf(tf, "%.3f\n", elapsed);
			fclose(tf);
		}
	}
	printf("Time: %.3f\n", elapsed);

	/* phase 6: print caption to stdout. */
	if (output[0])
	{
		printf("%s\n", output);
	}

	/* phase 6b: free caller-owned raw pixel buffer. */
	free(raw_rgb);

	/* phase 7/8: shutdown workers, destroy context, unload DLL. */
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
