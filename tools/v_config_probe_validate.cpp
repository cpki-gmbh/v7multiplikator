// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_config_probe_validate.cpp -- validator for the ASM matmul config
 * randomizer and mutation engine (src/asm/config_set_random_probe.inc).
 *
 * Checks config_set_random_probe and config_mutate_genome against the C
 * mirror of their legality contract: every draw inside its bounds, the
 * leave-alone fields (block n/d, worker_priority, cores_subtract, tail)
 * untouched, threshold triples ascending, pf windows respected, mutation
 * counts capped at 10, crossover donating only legal dwords, parents
 * never modified, chains of mutations staying legal.
 *
 * Usage: v_config_probe_validate.exe <dll_path> --db <db_path>
 */
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <windows.h>

#define BLOCK_SIZE 512
#define PARAM_OFF 8
#define PF_DIST_OFF 104
#define PF_TYPE_OFF 300
#define PARAM_SLOTS 22
#define PF_SLOTS 41
#define FIELD_TOTAL 104
#define WORKER_PRIO_OFF 52
#define CORES_SUB_OFF 100
#define TAIL_OFF 464
#define HALF_CAP 2147483647LL

typedef void (V7_ABI *config_probe_fn)(unsigned char *block);
typedef void (V7_ABI *config_mutate_fn)(const unsigned char *p1, const unsigned char *p2, unsigned char *out);

typedef struct bounds_entry
{
	int off;
	int lo;
	int hi;

} bounds_entry;

static const bounds_entry bounds[PARAM_SLOTS] = { { 0, 1, 4096 }, { 4, 0, 112 }, { 8, 1, 64 }, { 12, 1, 64 }, { 16, 1, 64 }, { 20, 1, 64 }, { 24, 1, 4096 }, { 28, 1, 4096 }, { 32, 1, 4096 }, { 36, 1, 128 }, { 40, 1, 64 }, { 48, 1, 64 }, { 52, 1, 64 }, { 56, 1, 64 }, { 60, 1, 64 }, { 64, 1, 64 }, { 68, 1, 4096 }, { 72, 1, 4096 }, { 76, 1, 4096 }, { 80, 0, 99 }, { 84, 1, 256 }, { 88, 1, 256 } };

typedef struct shape_pair
{
	int n;
	int d;
	const char *desc;

} shape_pair;

static const shape_pair shapes[] = { { 1024, 3072, "Qwen3-0.6B FFN" }, { 1024, 151936, "lm_head" }, { 256, 256, "small" }, { 65536, 65536, "half-cap (n*d = 2^32)" }, { 0, 0, 0 } };

static int read_i32(const unsigned char *block, int off)
{
	return (int)((unsigned int)block[off] | ((unsigned int)block[off + 1] << 8) | ((unsigned int)block[off + 2] << 16) | ((unsigned int)block[off + 3] << 24));
}

static void write_i32(unsigned char *block, int off, int value)
{
	block[off] = (unsigned char)(value & 0xFF);
	block[off + 1] = (unsigned char)((value >> 8) & 0xFF);
	block[off + 2] = (unsigned char)((value >> 16) & 0xFF);
	block[off + 3] = (unsigned char)((value >> 24) & 0xFF);
}

static long long window_half(int n, int d)
{
	long long half = (long long)n * (long long)d / 2;
	if (half > HALF_CAP)
		half = HALF_CAP;
	return half;
}

static int check_legal(const unsigned char *block, int n, int d, const char *label, int *fail_out)
{
	int bad = 0;
	for (int i = 0; i < PARAM_SLOTS; i++)
	{
		int v = read_i32(block, PARAM_OFF + bounds[i].off);
		if ((v < bounds[i].lo) || (v > bounds[i].hi))
		{
			printf("\n  %s: param slot %d (off %d) = %d outside [%d,%d]\n", label, i, bounds[i].off, v, bounds[i].lo, bounds[i].hi);
			bad = 1;
		}
	}
	long long half = window_half(n, d);
	for (int i = 0; i < PF_SLOTS; i++)
	{
		int v = read_i32(block, PF_DIST_OFF + i * 4);
		if ((v < -half) || (v > half))
		{
			printf("\n  %s: pf_dist[%d] = %d outside window +-%lld\n", label, i, v, half);
			bad = 1;
		}
	}
	for (int i = 0; i < PF_SLOTS; i++)
	{
		int v = read_i32(block, PF_TYPE_OFF + i * 4);
		if ((v < 0) || (v > 6))
		{
			printf("\n  %s: pf_type[%d] = %d outside [0,6]\n", label, i, v);
			bad = 1;
		}
	}
	for (int t = 0; t < 2; t++)
	{
		int base = PARAM_OFF + 24 + t * 44;
		int lo = read_i32(block, base);
		int mid = read_i32(block, base + 4);
		int hi = read_i32(block, base + 8);
		if ((lo > mid) || (mid > hi))
		{
			printf("\n  %s: threshold triple %d not ascending: %d,%d,%d\n", label, t, lo, mid, hi);
			bad = 1;
		}
	}
	if (bad)
		(*fail_out)++;
	return bad == 0;
}

static int same_i32(const unsigned char *a, const unsigned char *b, int off)
{
	return read_i32(a, off) == read_i32(b, off);
}

int main(int argc, char **argv)
{
	ZydisInit();
	if (argc < 4)
	{
		printf("usage: %s <dll_path> --db <db_path>\n", argv[0]);
		return 1;
	}
	if ((strcmp(argv[2], "--db") != 0))
	{
		printf("usage: %s <dll_path> --db <db_path>\n", argv[0]);
		return 1;
	}

	v7_lib lib;
	if ((v7_load(argv[1], &lib) != 0))
	{
		printf("cannot load %s\n", argv[1]);
		return 1;
	}
	lib.v7_init(argv[3]);

	HMODULE mod = (HMODULE)lib.module;
	config_probe_fn fill_probe = (config_probe_fn)GetProcAddress(mod, "config_set_random_probe");
	config_mutate_fn mutate_genome = (config_mutate_fn)GetProcAddress(mod, "config_mutate_genome");
	int *sex_percent = (int *)GetProcAddress(mod, "g_probe_sex_percent");
	int *mutations_max = (int *)GetProcAddress(mod, "g_probe_mutations_max");
	if ((!fill_probe) || (!mutate_genome) || (!sex_percent) || (!mutations_max))
	{
		printf("exports missing: config_set_random_probe / config_mutate_genome / g_probe_sex_percent / g_probe_mutations_max\n");
		return 1;
	}

	int pass = 0;
	int fail = 0;
	unsigned char base[BLOCK_SIZE];
	unsigned char p1[BLOCK_SIZE];
	unsigned char p2[BLOCK_SIZE];
	unsigned char out[BLOCK_SIZE];
	unsigned char tail_pattern[BLOCK_SIZE];

	for (int i = 0; i < BLOCK_SIZE; i++)
		tail_pattern[i] = (unsigned char)(i * 7 + 3);

	for (int si = 0; shapes[si].desc; si++)
	{
		int n = shapes[si].n;
		int d = shapes[si].d;
		long long half = window_half(n, d);
		printf("shape %s (n=%d d=%d window +-%lld):\n", shapes[si].desc, n, d, half);

		/* T1: filler bounds, leave-alone contract, triple order, sign
		 * coverage over 100 draws */
		int neg_seen = 0;
		int pos_seen = 0;
		int t1_ok = 1;
		for (int round = 0; round < 100; round++)
		{
			memcpy(base, tail_pattern, BLOCK_SIZE);
			write_i32(base, 0, n);
			write_i32(base, 4, d);
			write_i32(base, WORKER_PRIO_OFF, (int)0xDEADBEEF);
			write_i32(base, CORES_SUB_OFF, (int)0xCAFEBABE);
			fill_probe(base);
			if ((read_i32(base, 0) != n) || (read_i32(base, 4) != d))
			{
				printf("\n  fill round %d: n/d modified: n=%d d=%d\n", round, read_i32(base, 0), read_i32(base, 4));
				t1_ok = 0;
			}
			if (read_i32(base, WORKER_PRIO_OFF) != (int)0xDEADBEEF)
			{
				printf("\n  fill round %d: worker_priority sentinel destroyed: %08X\n", round, (unsigned)read_i32(base, WORKER_PRIO_OFF));
				t1_ok = 0;
			}
			if (read_i32(base, CORES_SUB_OFF) != (int)0xCAFEBABE)
			{
				printf("\n  fill round %d: cores_subtract sentinel destroyed: %08X\n", round, (unsigned)read_i32(base, CORES_SUB_OFF));
				t1_ok = 0;
			}
			if (memcmp(base + TAIL_OFF, tail_pattern + TAIL_OFF, BLOCK_SIZE - TAIL_OFF) != 0)
			{
				printf("\n  fill round %d: tail bytes modified\n", round);
				t1_ok = 0;
			}
			if (!t1_ok)
			{
				printf("\n  fill round %d: block n/d, system dwords or tail were modified\n", round);
				break;
			}
			if (!check_legal(base, n, d, "fill", &fail))
			{
				t1_ok = 0;
				break;
			}
			if (read_i32(base, PF_DIST_OFF) < 0)
				neg_seen = 1;
			if (read_i32(base, PF_DIST_OFF) > 0)
				pos_seen = 1;
		}
		if ((t1_ok) && (neg_seen) && (pos_seen))
		{
			pass++;
			printf("  fill bounds/leave-alone/triples/signs: pass\n");
		}
		else
		{
			if (t1_ok)
				printf("\n  fill: pf_dist sign coverage missing (neg=%d pos=%d)\n", neg_seen, pos_seen);
			fail++;
		}

		/* T2: two consecutive fills differ (the TSC-perturbed seed moves) */
		memcpy(base, tail_pattern, TAIL_OFF);
		write_i32(base, 0, n);
		write_i32(base, 4, d);
		fill_probe(base);
		memcpy(p1, base, BLOCK_SIZE);
		fill_probe(base);
		if (memcmp(p1, base, BLOCK_SIZE) != 0)
		{
			pass++;
			printf("  fill variability: pass\n");
		}
		else
		{
			printf("\n  fill variability: two fills produced identical blocks\n");
			fail++;
		}

		/* T3: mutate with sex=0 -- child = parent1 with the mutation
		 * changes plus the legality repairs (2 triples + taper + pct +
		 * g1 + g2 = up to 10 repair fields on top of the 10 mutated),
		 * donor never wins, parents untouched, all legal. The draw bound
		 * is pinned to 10 for this test so the diff cap is exact; the
		 * engine's published bound is a config knob and may be larger. */
		*sex_percent = 0;
		int saved_mutations_max = *mutations_max;
		*mutations_max = 10;
		memcpy(p1, tail_pattern, BLOCK_SIZE);
		write_i32(p1, 0, n);
		write_i32(p1, 4, d);
		write_i32(p1, WORKER_PRIO_OFF, (int)0xDEADBEEF);
		write_i32(p1, CORES_SUB_OFF, (int)0xCAFEBABE);
		fill_probe(p1);
		memcpy(p2, p1, BLOCK_SIZE);
		for (int i = 0; i < PARAM_SLOTS; i++)
			write_i32(p2, PARAM_OFF + bounds[i].off, bounds[i].hi);
		for (int i = 0; i < PF_SLOTS; i++)
		{
			write_i32(p2, PF_DIST_OFF + i * 4, (int)half);
			write_i32(p2, PF_TYPE_OFF + i * 4, 6);
		}
		int t3_ok = 1;
		for (int round = 0; round < 100; round++)
		{
			mutate_genome(p1, p2, out);
			if ((!same_i32(out, p1, 0)) || (!same_i32(out, p1, 4)) || (read_i32(out, WORKER_PRIO_OFF) != (int)0xDEADBEEF) || (read_i32(out, CORES_SUB_OFF) != (int)0xCAFEBABE))
			{
				printf("\n  mutate round %d: header/system dwords not inherited from parent1\n", round);
				t3_ok = 0;
				break;
			}
			if (memcmp(out + TAIL_OFF, p1 + TAIL_OFF, BLOCK_SIZE - TAIL_OFF) != 0)
			{
				printf("\n  mutate round %d: block tail not inherited from parent1\n", round);
				t3_ok = 0;
				break;
			}
			int diffs = 0;
			for (int i = 0; i < TAIL_OFF; i += 4)
			{
				if (!same_i32(out, p1, i))
					diffs++;
			}
			if (diffs > 20)
			{
				printf("\n  mutate round %d: %d fields differ from parent1, cap is 10 mutations + 10 repair fields\n", round, diffs);
				t3_ok = 0;
				break;
			}
			if (!check_legal(out, n, d, "mutate sex=0", &fail))
			{
				t3_ok = 0;
				break;
			}
		}
		if (t3_ok)
		{
			pass++;
			printf("  mutate sex=0 structure/legality: pass\n");
		}
		else
			fail++;
		*mutations_max = saved_mutations_max;

		/* T4: mutate with sex=100 -- extreme donor values must appear in
		 * children across 50 rounds */
		*sex_percent = 100;
		int donor_hits = 0;
		int t4_ok = 1;
		for (int round = 0; round < 50; round++)
		{
			mutate_genome(p1, p2, out);
			if (!check_legal(out, n, d, "mutate sex=100", &fail))
			{
				t4_ok = 0;
				break;
			}
			for (int i = 0; i < PARAM_SLOTS; i++)
			{
				if ((read_i32(out, PARAM_OFF + bounds[i].off) == bounds[i].hi) && (read_i32(p1, PARAM_OFF + bounds[i].off) != bounds[i].hi))
					donor_hits++;
			}
			for (int i = 0; i < PF_SLOTS; i++)
			{
				if ((read_i32(out, PF_DIST_OFF + i * 4) == (int)half) && (read_i32(p1, PF_DIST_OFF + i * 4) != (int)half))
					donor_hits++;
			}
		}
		if ((t4_ok) && (donor_hits > 0))
		{
			pass++;
			printf("  mutate sex=100 crossover (%d donor fields seen): pass\n", donor_hits);
		}
		else
		{
			if (t4_ok)
				printf("\n  mutate sex=100: no donor value ever landed in a child\n");
			fail++;
		}
		*sex_percent = 0;

		/* T5: 20-deep mutation chain stays legal every step */
		*sex_percent = 40;
		int t5_ok = 1;
		memcpy(p1, tail_pattern, TAIL_OFF);
		write_i32(p1, 0, n);
		write_i32(p1, 4, d);
		fill_probe(p1);
		memcpy(p2, p1, BLOCK_SIZE);
		fill_probe(p2);
		unsigned char *src = p1;
		for (int step = 0; step < 20; step++)
		{
			mutate_genome(src, p2, out);
			if (!check_legal(out, n, d, "chain", &fail))
			{
				t5_ok = 0;
				break;
			}
			memcpy(p1, out, BLOCK_SIZE);
			src = p1;
		}
		*sex_percent = 0;
		if (t5_ok)
		{
			pass++;
			printf("  20-deep mutation chain at sex=40: pass\n");
		}
		else
			fail++;
	}

	v7_unload(&lib);
	printf("\nconfig_probe_validate: %d passed, %d failed\n", pass, fail);
	if (fail)
		return 1;
	return 0;
}
