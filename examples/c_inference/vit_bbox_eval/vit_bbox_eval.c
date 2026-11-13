// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* vit_bbox_eval.c -- vi_t bounding-box evaluation harness.
 *
 * synthesizes a solid-color background image in C, draws a red circle and a
 * green square into the pixel buffer, then asks the loaded vision model to
 * return bounding boxes for both shapes.
 *
 * this is an EVALUATION tool, not a unit test: there is no pass/fail assertion
 * on the returned box quality. the only requirement is that the model ANSWERS
 * (produce a non-empty response) -- the harness exits 0 as long as it did.
 *
 * run it once per model database, e.g.:
 *   vit_bbox_eval.exe bin\v7_threading.dll weights\databases\qwen3_vl_4b\lz4ht.db
 *   vit_bbox_eval.exe bin\v7_threading.dll weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db -q8 0
 *
 * usage: vit_bbox_eval.exe <dll_path> <db_path> [-q8 <0|1>] [prompt] [cx] [cy] [r] [sqx] [sqy] [sqs]
 *   dll_path -- path to v7_threading.dll (REQUIRED, no default)
 *   db_path -- path to lz4ht.db (REQUIRED, no default)
 *   -q8     -- quantization mode: 0 = F32, 1 = Q8 (default: 1)
 *   prompt  -- text prompt (default: locate red circle + green square)
 *   cx,cy,r -- red circle center x/y + radius in pixels (default 180 180 90)
 *   sqx,sqy,sqs -- green square top-left x/y + side in pixels (default 400 400 140)
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OUTPUT_BUF_SIZE 32768

#define IMG_W 640
#define IMG_H 640

/* draw a filled disc (center cx,cy radius r) in the RGB buffer. */
void draw_filled_circle(unsigned char *rgb, int w, int h, int cx, int cy, int r, unsigned char cr, unsigned char cg, unsigned char cb)
{
	int y0 = cy - r;
	int y1 = cy + r;
	if (y0 < 0)
		y0 = 0;
	if (y1 > h - 1)
		y1 = h - 1;
	for (int y = y0; y <= y1; y++)
	{
		int dy = y - cy;
		int dx = (int)sqrtf((float)(r * r - dy * dy));
		int x0 = cx - dx;
		int x1 = cx + dx;
		if (x0 < 0)
			x0 = 0;
		if (x1 > w - 1)
			x1 = w - 1;
		for (int x = x0; x <= x1; x++)
		{
			unsigned char *p = rgb + ((int)y * w + (int)x) * 3u;
			p[0] = cr;
			p[1] = cg;
			p[2] = cb;
		}
	}
}

/* draw a filled axis-aligned square (top-left x0,y0, side s) in the buffer. */
void draw_filled_square(unsigned char *rgb, int w, int h, int x0, int y0, int s, unsigned char cr, unsigned char cg, unsigned char cb)
{
	int x1 = x0 + s - 1;
	int y1 = y0 + s - 1;
	if (x0 < 0)
		x0 = 0;
	if (y0 < 0)
		y0 = 0;
	if (x1 > w - 1)
		x1 = w - 1;
	if (y1 > h - 1)
		y1 = h - 1;
	for (int y = y0; y <= y1; y++)
	{
		for (int x = x0; x <= x1; x++)
		{
			unsigned char *p = rgb + ((int)y * w + (int)x) * 3u;
			p[0] = cr;
			p[1] = cg;
			p[2] = cb;
		}
	}
}

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	/* split args into positionals and -q8 <0|1> (flag may appear anywhere). */
	char *pos[9];
	int npos = 0;
	int q8 = 1; /* default Q8 (continuity with the original harness behavior) */
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-q8") == 0 && i + 1 < argc)
		{
			q8 = atoi(argv[++i]);
			continue;
		}
		if (npos < 9)
			pos[npos++] = argv[i];
	}
	if (npos < 2 || q8 < 0 || q8 > 1)
	{
		printf("Usage: vit_bbox_eval.exe <dll_path> <db_path> [-q8 <0|1>] [prompt] [cx] [cy] [r] [sqx] [sqy] [sqs]\n");
		printf("  dll_path is REQUIRED -- path to v7_threading.dll (no default)\n");
		printf("  db_path is REQUIRED -- path to lz4ht.db (no default)\n");
		return 1;
	}

	char *dll_path = pos[0];
	char *db_path = pos[1];
	char *prompt = (npos > 2) ? pos[2]
	                          : "This image contains a red circle and a green square. "
	                            "Locate each shape and provide its bounding box coordinates.";
	int cx = (npos > 3) ? atoi(pos[3]) : 180;  /* circle center x */
	int cy = (npos > 4) ? atoi(pos[4]) : 180;  /* circle center y */
	int r = (npos > 5) ? atoi(pos[5]) : 90;    /* circle radius   */
	int sqx = (npos > 6) ? atoi(pos[6]) : 400; /* square top-left x */
	int sqy = (npos > 7) ? atoi(pos[7]) : 400; /* square top-left y */
	int sqs = (npos > 8) ? atoi(pos[8]) : 140; /* square side       */

	printf("loading: %s (mode: %s)\n", db_path, q8 ? "Q8" : "F32");
	printf("question: %s\n", prompt);

	/* phase 2: load v7_threading.dll + resolve exports. */
	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("[FATAL] LoadLibraryA failed for %s\n", dll_path);
		return 1;
	}

	/* phase 3: initialize database. */
	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] V7_init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] V7_init failed\n");
		v7_unload(&dll);
		return 1;
	}

	/* phase 4: initialize inference (context hidden in BSS). Sec.172: a gated
	 * entry -- a caught FATAL also returns 0, so the real return AND the death
	 * flag must both be checked. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}

	/* phase 5: synthesize the test image. background light gray, red circle in
	 * the upper-left area, green square in the lower-right area. */
	unsigned char *rgb = (unsigned char *)malloc((int)IMG_W * (int)IMG_H * 3u);
	if (!rgb)
	{
		printf("[FATAL] malloc failed for pixel buffer\n");
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	memset(rgb, 0xE0, (int)IMG_W * (int)IMG_H * 3u); /* light gray bg */
	draw_filled_circle(rgb, IMG_W, IMG_H, cx, cy, r, 220, 30, 30);
	draw_filled_square(rgb, IMG_W, IMG_H, sqx, sqy, sqs, 30, 180, 60);

	printf("synthesized %dx%d image (red circle @ (%d,%d) r%d, "
	       "green square @ (%d,%d) %dx%d)\n",
	       IMG_W, IMG_H, cx, cy, r, sqx, sqy, sqs, sqs);

	/* phase 6: run the vision chat, print the model's answer. */
	char output[OUTPUT_BUF_SIZE];
	memset(output, 0, sizeof(output));
	dll.v7_chat_image(prompt, 0, rgb, IMG_W, IMG_H, output, OUTPUT_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died during vision chat: %s\n", dll.v7_get_death_reason());
		free(rgb);
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}

	if (output[0])
	{
		printf("%s\n", output);
	}

	free(rgb);
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);

	/* evaluation semantics: the model only has to answer. */
	if (!output[0])
	{
		printf("[FAIL] model produced no answer\n");
		return 1;
	}
	printf("[OK] model answered\n");
	return 0;
}
