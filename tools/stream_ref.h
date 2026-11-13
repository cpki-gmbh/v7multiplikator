// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* stream_ref.h -- StreamingLLM reference model (oracle; test-only, do NOT
 * include from the engine). Caller validates S >= 1, W >= 1, C <= seq_len. */

#pragma once

typedef struct { int lo, hi; } stream_rng;     /* inclusive slot range; hi < lo = empty */

static int stream_slot(int pos, int S, int W, int C)
{
	return (pos < C) ? pos : S + (pos - S) % W;
}

typedef struct {
	stream_rng fresh, sinks, stale;            /* slot ranges */
	int q_fresh, q_sink, q_stale;
	int h;                                     /* self slot == the RoPE position of this token's K */
} stream_plan;

static stream_plan stream_plan_for(int pos, int S, int W, int C)
{
	stream_plan p;
	p.h = stream_slot(pos, S, W, C);
	p.fresh.lo = S;
	p.fresh.hi = p.h;
	p.sinks.lo = 0;
	p.sinks.hi = S - 1;
	p.stale.lo = p.h + 1;
	p.stale.hi = C - 1;                        /* empty when h == C-1 */
	p.q_fresh = p.h;
	p.q_sink = C - 1;
	p.q_stale = p.h + W;                       /* delta2 = h - S + 1 (never indexed directly) */
	return p;
}

/* Distance for a cached row at slot o (paper cache-relative distance).
 * Returns -1 for an invalid slot or a pre-wrap pos. */
static int stream_expected_distance(int pos, int o, int S, int W, int C)
{
	stream_plan p;
	if ((pos < C) || (o < 0) || (o >= C))
		return -1;
	p = stream_plan_for(pos, S, W, C);
	if (o < S)
		return p.q_sink - o;                   /* sink row          */
	if (o <= p.h)
		return p.q_fresh - o;                  /* fresh window row  */
	return p.q_stale - o;                      /* stale window row  */
}
