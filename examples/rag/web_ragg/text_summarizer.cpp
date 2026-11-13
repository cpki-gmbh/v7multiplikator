// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

// text_summarizer.cpp - LLM-based text summarization using v7_threading.dll
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "v7_lib.h"
#include "text_summarizer.h"

typedef void *(*runq_init_fn)(void);
typedef int (*runq_generate_fn)(void *, char *, char *, int);
typedef void (*runq_destroy_fn)(void *);

typedef struct
{
	HMODULE dll;
	void *runq_ctx;
	runq_init_fn init;
	runq_generate_fn generate;
	runq_destroy_fn destroy;
	struct v7_lib *v7;
	char error[256];
	int debug;
	FILE *log;
} summarizer_ctx;

void slog(summarizer_ctx *ctx, char *fmt, ...)
{
	if (!ctx)
		return;
	if (ctx->log)
	{
		va_list args;
		va_start(args, fmt);
		vfprintf(ctx->log, fmt, args);
		va_end(args);
		fflush(ctx->log);
	}
	if (ctx->debug)
	{
		va_list args;
		va_start(args, fmt);
		vprintf(fmt, args);
		va_end(args);
	}
}

summarizer_context summarizer_init(char *dll_path, char *ini_path)
{
	summarizer_ctx *ctx = (summarizer_ctx *)calloc(1, sizeof(summarizer_ctx));
	if (!ctx)
		return 0;
	ctx->debug = 1;
	if (ini_path)
	{
		ctx->log = fopen("summarizer.log", "a");
		char debug_val[8];
		if (GetPrivateProfileStringA("summarizer", "Debug", "1", debug_val, sizeof(debug_val), ini_path))
		{
			ctx->debug = atoi(debug_val);
		}
	}
	slog(ctx, "[SUMMARIZER] initializing (runq path)...\n");
	if (dll_path)
	{
		ctx->dll = LoadLibraryA(dll_path);
	}
	else
	{
		char exe_path[MAX_PATH];
		GetModuleFileNameA(0, exe_path, sizeof(exe_path));
		char *last = strrchr(exe_path, '\\');
		if (last)
			*last = '\0';
		if (strlen(exe_path) + strlen("\\runq.dll") >= MAX_PATH)
		{
			slog(ctx, "[SUMMARIZER] path too long for DLL\n");
			strcpy(ctx->error, "Path too long");
			free(ctx);
			return 0;
		}
		strcat_s(exe_path, MAX_PATH, "\\runq.dll");
		ctx->dll = LoadLibraryA(exe_path);
	}
	if (!ctx->dll)
	{
		slog(ctx, "[SUMMARIZER] failed to load DLL: %lu\n", GetLastError());
		strcpy(ctx->error, "Failed to load runq.dll");
		if (ctx->log)
			fclose(ctx->log);
		free(ctx);
		return 0;
	}
	ctx->init = (runq_init_fn)GetProcAddress(ctx->dll, "runq_init");
	ctx->generate = (runq_generate_fn)GetProcAddress(ctx->dll, "runq_generate");
	ctx->destroy = (runq_destroy_fn)GetProcAddress(ctx->dll, "runq_destroy");
	if (!ctx->init || !ctx->generate || !ctx->destroy)
	{
		slog(ctx, "[SUMMARIZER] failed to get DLL functions\n");
		strcpy(ctx->error, "Failed to get runq functions");
		FreeLibrary(ctx->dll);
		if (ctx->log)
			fclose(ctx->log);
		free(ctx);
		return 0;
	}
	ctx->runq_ctx = ctx->init();
	if (!ctx->runq_ctx)
	{
		slog(ctx, "[SUMMARIZER] failed to initialize runq\n");
		strcpy(ctx->error, "Failed to initialize runq");
		FreeLibrary(ctx->dll);
		if (ctx->log)
			fclose(ctx->log);
		free(ctx);
		return 0;
	}
	slog(ctx, "[SUMMARIZER] initialized successfully\n");
	return ctx;
}

summarizer_context summarizer_init_with_v7(char *ini_path, struct v7_lib *v7)
{
	if (!v7)
		return 0;
	summarizer_ctx *ctx = (summarizer_ctx *)calloc(1, sizeof(summarizer_ctx));
	if (!ctx)
		return 0;
	ctx->v7 = v7;
	ctx->debug = 1;
	if (ini_path)
	{
		ctx->log = fopen("summarizer.log", "a");
		char debug_val[8];
		if (GetPrivateProfileStringA("summarizer", "Debug", "1", debug_val, sizeof(debug_val), ini_path))
		{
			ctx->debug = atoi(debug_val);
		}
	}
	slog(ctx, "[SUMMARIZER] initializing with v7...\n");
	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real
	 * return AND the death flag must both be checked. */
	if ((!v7->v7_initialize_inference()) || v7->v7_engine_is_dead())
	{
		slog(ctx, "[SUMMARIZER] failed to create v7 inference context\n");
		snprintf(ctx->error, sizeof(ctx->error), "Failed to create v7 context: %s",
		         v7->v7_engine_is_dead() ? v7->v7_get_death_reason() : "init returned 0");
		if (ctx->log)
			fclose(ctx->log);
		free(ctx);
		return 0;
	}
	slog(ctx, "[SUMMARIZER] initialized with v7 successfully\n");
	return ctx;
}

void summarizer_shutdown(summarizer_context ctx_)
{
	summarizer_ctx *ctx = (summarizer_ctx *)ctx_;
	if (!ctx)
		return;
	if (ctx->v7)
	{
		ctx->v7->v7_free_inference();
	}
	else
	{
		if (ctx->runq_ctx)
		{
			ctx->destroy(ctx->runq_ctx);
		}
		if (ctx->dll)
		{
			FreeLibrary(ctx->dll);
		}
	}
	if (ctx->log)
	{
		fclose(ctx->log);
	}
	free(ctx);
}

int summarizer_summarize(summarizer_context ctx_, char *text, char *output, int max_len)
{
	return summarizer_summarize_extended(ctx_, text, max_len, output, max_len, "Summarize this text concisely, preserving key information:");
}

int summarizer_summarize_extended(summarizer_context ctx_, char *text, int max_len, char *output, int max_out, char *purpose)
{
	summarizer_ctx *ctx = (summarizer_ctx *)ctx_;
	if (!ctx || !text || !output)
		return SUMMARIZER_ERR;
	if (max_len <= 0 || max_out <= 0)
		return SUMMARIZER_ERR;
	if (strlen(text) < 100)
	{
		strncpy(output, text, max_out - 1);
		output[max_out - 1] = '\0';
		return SUMMARIZER_OK;
	}
	char prompt[8192];
	int prompt_len = snprintf(prompt, sizeof(prompt), "%s\n\n%s\n\nProvide a concise summary in %d characters or less:\n", purpose ? purpose : "Summarize this text concisely:", text, max_len);
	if (prompt_len < 0 || (int)prompt_len >= sizeof(prompt))
	{
		strcpy(ctx->error, "Text too long for summarization");
		return SUMMARIZER_ERR;
	}
	slog(ctx, "[SUMMARIZER] summarizing %d chars into %d chars...\n", (int)strlen(text), max_len);
	if (ctx->v7)
	{
		ctx->v7->v7_generate(prompt, output, max_out - 1, 0);
		if (ctx->v7->v7_engine_is_dead())
		{
			snprintf(ctx->error, sizeof(ctx->error), "Generation failed: %s", ctx->v7->v7_get_death_reason());
			return SUMMARIZER_ERR;
		}
	}
	else
	{
		int result = ctx->generate(ctx->runq_ctx, prompt, output, max_out - 1);
		if (result != 0)
		{
			slog(ctx, "[SUMMARIZER] generation failed: %d\n", result);
			strcpy(ctx->error, "Generation failed");
			return SUMMARIZER_ERR;
		}
	}
	slog(ctx, "[SUMMARIZER] summary generated: %d chars\n", (int)strlen(output));
	return SUMMARIZER_OK;
}

char *summarizer_error(summarizer_context ctx_)
{
	summarizer_ctx *ctx = (summarizer_ctx *)ctx_;
	if (!ctx)
		return "Null context";
	return ctx->error;
}
