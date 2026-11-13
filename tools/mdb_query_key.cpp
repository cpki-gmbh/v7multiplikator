// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * mdb_query_key.exe -- Look up a single MicroDB key and display value
 *
 * Usage: mdb_query_key.exe <dll_path> <key> --db <db_path>
 *
 * Shows type-aware output: Config, weight_meta, MemLayout, string, int32, float, hex dump
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "mdb_shared.h"
#include "mdb_cli.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv)
{
	char *dll_path = 0;
	char *db_path = 0;
	char *key = 0;
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--db") == 0 && i + 1 < argc)
		{
			db_path = argv[++i];
		}
		else if (!dll_path)
			dll_path = argv[i];
		else if (!key)
			key = argv[i];
	}
	if (!dll_path || !db_path || !key)
	{
		printf("usage: mdb_query_key.exe <dll_path> <key> --db <db_path>\n");
		return 1;
	}

	v7_lib lib;
	if (!mdb_cli_open(&lib, dll_path, db_path))
		return 1;

	unsigned __int64 out_len = 0;
	void *data = lib.ht_find(key, &out_len);
	if (!data)
	{
		printf("key not found: %s\n", key);
		v7_unload(&lib);
		return 1;
	}

	printf("key: %s\n", key);
	printf("length: %llu bytes\n", (unsigned long long)out_len);

	/* Special display for known system keys */
	if (strcmp(key, "__sys_config") == 0 && out_len >= 64)
	{
		int *p = (int *)data;
		printf("  Config (16 ints):\n");
		printf("    reserved0=%d reserved1=%d dim=%d hidden_dim=%d\n", p[0], p[1], p[2], p[3]);
		printf("    n_layers=%d n_heads=%d n_kv_heads=%d vocab_size=%d\n", p[4], p[5], p[6], p[7]);
		printf("    seq_len=%d head_dim=%d shared_cls=%d group_size=%d\n", p[8], p[9], p[10], p[11]);
		printf("    qk_norm=%d rope_base=%d\n", p[12], p[13]);
	}
	else if (strcmp(key, "__sys_mem_layout") == 0 && out_len >= 80)
	{
		int *p = (int *)data;
		printf("  MemLayout (20 ints):\n");
		for (int i = 0; i < 20; i++)
			printf("    [%2d] = %d\n", i, p[i]);
	}
	else if (strcmp(key, "__sys_vision_config") == 0 && out_len >= 640)
	{
		/* Expanded ViT config: 16 x int32 raw header verbatim + derived qwords
		 * (64-520) + cross-model feature fields (528-600). Byte offsets are the
		 * authoritative spec from db_build_tool.c build_vision_cfg() (must match
		 * inf_structs.inc VCFG_*). */
		int *hdr = (int *)data;
		__int64 *cfg = (__int64 *)data;
		printf("  Vision raw header (16 int32):\n");
		printf("    image_token=%d start=%d end=%d deepstack=%d vis_hidden=%d out_hidden=%d\n", hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]);
		printf("    patch=%d merge=%d temporal=%d pos_emb=%d ds_tap0=%d ds_tap1=%d ds_tap2=%d depth=%d inter=%d heads=%d\n", hdr[6], hdr[7], hdr[8], hdr[9], hdr[10], hdr[11], hdr[12], hdr[13], hdr[14], hdr[15]);
		printf("  Derived qwords (offset: value):\n");
		struct
		{
			int off;
			char *name;
		} names[] = {
			{ 64, "VCFG_HIDDEN" },
			{ 72, "VCFG_HIDDEN_F4" },
			{ 80, "VCFG_INTER" },
			{ 88, "VCFG_QKV_DIM" },
			{ 96, "VCFG_QKV_STRIDE" },
			{ 104, "VCFG_HEADS" },
			{ 112, "VCFG_HEAD_DIM" },
			{ 120, "VCFG_HEAD_DIM_F4" },
			{ 128, "VCFG_DEPTH" },
			{ 136, "VCFG_NUM_POS" },
			{ 144, "VCFG_NUM_POS_F4" },
			{ 152, "VCFG_GRID_SIDE" },
			{ 160, "VCFG_GRID_MINUS1_I" },
			{ 168, "VCFG_GRID_MINUS1_F" },
			{ 176, "VCFG_MERGE" },
			{ 184, "VCFG_MERGE_UNIT" },
			{ 192, "VCFG_PATCH_SIZE" },
			{ 200, "VCFG_PATCH_SIZE_F4" },
			{ 208, "VCFG_OUT_HIDDEN" },
			{ 216, "VCFG_PATCH_DIM" },
			{ 224, "VCFG_PATCH_DIM_F4" },
			{ 232, "VCFG_PACKED_DIM" },
			{ 240, "VCFG_PACKED_DIM_F4" },
			{ 248, "VCFG_N_DEEPSTACK" },
			{ 256, "VCFG_ROPE_QTR" },
			{ 264, "VCFG_ROPE_HALF" },
			{ 272, "VCFG_ROPE_HALF_F4" },
			{ 280, "VCFG_DS_TAP0" },
			{ 288, "VCFG_DS_TAP1" },
			{ 296, "VCFG_DS_TAP2" },
			{ 304, "VCFG_L_QKV_W" },
			{ 312, "VCFG_L_QKV_B" },
			{ 320, "VCFG_L_PROJ_W" },
			{ 328, "VCFG_L_PROJ_B" },
			{ 336, "VCFG_L_FC1_W" },
			{ 344, "VCFG_L_FC1_B" },
			{ 352, "VCFG_L_FC2_W" },
			{ 360, "VCFG_L_FC2_B" },
			{ 368, "VCFG_BLK_NORM1_STRIDE" },
			{ 376, "VCFG_BLK_QKV_W_STRIDE" },
			{ 384, "VCFG_BLK_QKV_B_STRIDE" },
			{ 392, "VCFG_BLK_PROJ_W_STRIDE" },
			{ 400, "VCFG_BLK_PROJ_B_STRIDE" },
			{ 408, "VCFG_BLK_FC1_W_STRIDE" },
			{ 416, "VCFG_BLK_FC1_B_STRIDE" },
			{ 424, "VCFG_BLK_FC2_W_STRIDE" },
			{ 432, "VCFG_BLK_FC2_B_STRIDE" },
			{ 440, "VCFG_MG_FC1_W_STRIDE" },
			{ 448, "VCFG_MG_FC1_B_STRIDE" },
			{ 456, "VCFG_MG_FC2_W_STRIDE" },
			{ 464, "VCFG_MG_FC2_B_STRIDE" },
			{ 472, "VCFG_DS_FC1_W_STRIDE" },
			{ 480, "VCFG_DS_FC1_B_STRIDE" },
			{ 488, "VCFG_DS_FC2_W_STRIDE" },
			{ 496, "VCFG_DS_FC2_B_STRIDE" },
			{ 504, "VCFG_PE_W_STRIDE" },
			{ 512, "VCFG_POS_EMBED_BYTES" },
			{ 520, "VCFG_ATTN_SCALE" },
			{ 528, "VCFG_NORM_TYPE" },
			{ 536, "VCFG_FFN_TYPE" },
			{ 544, "VCFG_HAS_CLS" },
			{ 552, "VCFG_FEATURE_SELECT" },
			{ 560, "VCFG_PROJ_HIDDEN" },
			{ 568, "VCFG_PROJ_LAYERS" },
			{ 576, "VCFG_PROJ_ACTIVATION" },
			{ 584, "VCFG_POS_EMBED_TYPE" },
			{ 592, "VCFG_IMAGE_SIZE" },
			{ 600, "VCFG_DEPTH_TAPS" },
		};
		for (int i = 0; i < sizeof(names) / sizeof(names[0]); i++)
		{
			int o = names[i].off;
			printf("    %-24s [%3d] = %lld", names[i].name, o, (long long)cfg[o / 8]);
			if (o == 168)
				printf("  (float %g)", *(float *)((unsigned char *)data + o));
			if (o == 520)
				printf("  (1/sqrt(head_dim) float %g)", *(float *)((unsigned char *)data + o));
			printf("\n");
		}
	}
	else
	{
		mdb_print_value_typed((unsigned char *)data, out_len);
	}

	v7_unload(&lib);
	return 0;
}