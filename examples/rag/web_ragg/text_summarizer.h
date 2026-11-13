// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

// text_summarizer.h - LLM-based text summarization using v7_threading.dll
#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#ifdef __cplusplus
extern "C"
{
#endif

	struct v7_lib;

#define SUMMARIZER_OK 0
#define SUMMARIZER_ERR -1
#define SUMMARIZER_NOT_INIT -2

	typedef void *summarizer_context;

	summarizer_context summarizer_init(char *dll_path, char *ini_path);
	summarizer_context summarizer_init_with_v7(char *ini_path, struct v7_lib *v7);
	void summarizer_shutdown(summarizer_context ctx);

	int summarizer_summarize(summarizer_context ctx, char *text, char *output, int max_len);
	int summarizer_summarize_extended(summarizer_context ctx, char *text, int max_len, char *output, int max_out, char *purpose);

	char *summarizer_error(summarizer_context ctx);

#ifdef __cplusplus
}
#endif
