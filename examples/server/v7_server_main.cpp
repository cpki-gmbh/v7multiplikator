// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v7_server_main.cpp — OpenAI-compatible Chat Completions API server + browser
 * chat (entry point of the v7_server_*.cpp source split)
 *
 * HTTP server on two threads -- one inference/response thread plus one
 * health/accept thread handing connections over via a queue -- that wraps
 * the engine library's ASM inference
 * pipeline (v7s_inference_create / v7s_chat / v7s_generate / v7s_chat_image_mem
 * / v7s_inference_destroy) behind the standard OpenAI /v1/chat/completions and
 * /v1/completions endpoints, plus:
 *
 *   GET  /                     browser chat UI (chat + image chat, persistent KV)
 *   POST /v1/chat/completions  multi-turn text chat (server-side session state)
 *   POST /v1/completions       single-turn generate
 *   POST /v1/vision/chat       image turn on the SAME chat session (base64 image)
 *   POST /v1/chat/reset        start a fresh session (clears the KV cache)
 *   GET/POST /quit?code=N      terminate, exit code N (default 42 = clean quit;
 *                              the server.bat restart loop stops only on 42)
 *   GET/POST /sidecar?<path>   load a SectorLogo sidecar at runtime (the query
 *                              string IS the fitle path, verbatim)
 *   GET  /v1/health, /v1/models
 *
 * v7s_chat has a 6th reset parameter (1 = new session, 0 = continue).
 * reset=1 is used on the first turn after /v1/chat/reset or server start;
 * every later turn uses reset=0 so the KV cache persists across turns
 * (clients send only the NEW user message, not the whole history).
 *
 * Vision: image_b64 is decoded to raw 24bpp RGB (GDI+ on Windows, the
 * vendored stb_image on Linux) and passed to v7s_chat_image_mem — the image
 * turn continues the same chat session.
 *
 * Usage: v7_server <library> <lz4ht.db> <port> <model_id> [flags]
 *
 * Platform libraries: Windows links ws2_32, gdiplus, ole32; Linux builds
 * from the same sources through the Makefile (no GDI+).
 */

#include "v7_server_common.h"

/* Zydis VEH crash disassembly: a crash in the DLL (or in this process) is
 * disassembled at the fault site with a register dump instead of dying as
 * an opaque exit code. The build already links Zydis.lib and puts the
 * project root on the include path. */
#include "zydis_debug_header.h"

/* ========================================================================
 * Globals -- the DEFINITIONS. v7_server_common.h declares all of them
 * extern for the other server translation units.
 * ======================================================================== */

struct v7_lib server_dll;
std::atomic<int> server_running{ 1 };
std::atomic<int> request_counter{ 0 };
std::atomic<int> reset_next_turn{ 1 };  /* 1 = next chat turn starts a fresh session */
std::atomic<int> server_exit_code{ 0 }; /* set by /quit — returned from main() */
int hook_enabled = 0;                   /* sidecar hook enabled via /sidecar */
int hook_executed = 0;                  /* an inference turn completed with hook on */
int g_server_chat_turn = 0;             /* sidecar turn counter: 0 on the reset
                                           turn, +1 per continuation — written
                                           to the microdb as
                                           "__server_chat__turn" before every
                                           chat turn so SectorLogo sidecars can
                                           gate on the first turn */
int g_max_gen = 0;                      /* server-wide default generation budget:
                                           0 = UNLIMITED (the engine stops at EOS
                                           or the output buffer -- the only legal
                                           cap is the buffer). Set from the
                                           -maxgen CLI flag; a request's own
                                           max_tokens still wins when > 0 */
#ifdef _WIN32
static ULONG_PTR g_gdiplus_token;
static HRESULT com_hr;
#endif

/* server-side conversation history: malloc'd -history entries, evicts the
 * oldest pair when full (the array lives here; web_render owns the append/
 * clear logic) */
chat_msg *g_chat_history = 0;
int g_chat_history_cap = 0;

static void server_platform_cleanup()
{
#ifdef _WIN32
	if (g_gdiplus_token)
		Gdiplus::GdiplusShutdown(g_gdiplus_token);
	if (SUCCEEDED(com_hr))
		CoUninitialize();
#endif
}

static void server_network_cleanup()
{
#ifdef _WIN32
	WSACleanup();
#endif
}

#ifdef __linux__
static void *linux_health_thread(void *argument)
{
	health_thread(argument);
	return nullptr;
}
#endif
int g_engine_thinking = 1;              /* what WE last set via set_engine_thinking —
                                           the rescue restores THIS, never a literal 1 */
float g_sampler_temp = 0.6f;            /* the Qwen3 model-card sampling default (thinking mode: temp 0.6);
                                           every set_engine_thinking re-arms the mode's pair, the think-tag
                                           rescue saves/restores around its ladder */

int background_image_name(char *out, int out_cap)
{
	unsigned __int64 value_len = 0;
	char *value = (char *)server_dll.v7_db_find("__server_html_background_image", &value_len);
	if (!value || value_len == 0)
	{
		out[0] = 0;
		return 0;
	}
	int name_len = (int)((value_len < (unsigned __int64)(out_cap - 1)) ? value_len : (out_cap - 1));
	memcpy(out, value, name_len);
	out[name_len] = 0;
	return out[0] != 0;
}

/* Chat avatar image: the sidecar-set persona picture shown on the bot's
 * messages. Same contract as the background key -- an empty or missing
 * value means "no avatar". */
int avatar_image_name(char *out, int out_cap)
{
	unsigned __int64 value_len = 0;
	char *value = (char *)server_dll.v7_db_find("__server__chat_avatar_image", &value_len);
	if (!value || value_len == 0)
	{
		out[0] = 0;
		return 0;
	}
	int name_len = (int)((value_len < (unsigned __int64)(out_cap - 1)) ? value_len : (out_cap - 1));
	memcpy(out, value, name_len);
	out[name_len] = 0;
	return out[0] != 0;
}

int main(int argc, char *argv[])
{
	ZydisInit(); /* the VEH crash-disassembly handler must be first: a crash anywhere (engine load, inference, socket loop) gets a fault-site disassembly */
	int worker_prio = -15; /* default: idle -- the PC keeps working while the server runs */
	int server_stream = 0; /* -stream: 1 = every chat session runs the StreamingLLM ring KV cache */
	int server_sinks = 4;  /* -sinks: attention-sink count S (kept prefix, never evicted) */
	int server_window = 996; /* -window: sliding window W (fill grade C = S + W) */
	int server_history = 4096; /* -history: server-side web chat entry capacity (pairs evicted when full) */
	/* Positional args (dll, db, port, model) must SKIP the flag pairs:
	 * "server dll db 8080 -prio 0" would otherwise read "-prio" as the
	 * model_id. A bad argument value prints and EXITS: a silently
	 * corrected flag would serve with configuration the operator never
	 * asked for. */
	char *positionals[4] = { 0, 0, 0, 0 };
	int num_positionals = 0;
	for (int i = 1; i < argc; i++)
	{
		/* A help/version request is an operator probing the CLI, not a
		 * configuration error: print the usage and exit 0. The engine
		 * version is printed by the startup banner once the library is
		 * loaded, so this path needs no engine load. */
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "--version") == 0)
		{
			printf("v7 server\n\n"
			       "Usage: %s <library> <lz4ht.db> <port> <model_id> [-prio N] [-maxgen N] [-history N]\n"
			       "                 [-stream 0|1] [-sinks N] [-window N]\n", argv[0]);
			return 0;
		}
		if (strcmp(argv[i], "-prio") == 0)
		{
			/* -prio is ALWAYS the flag, even as the last argument -- a bare
			 * "-prio" must never be swallowed as a positional (DLL path). */
			if (i + 1 >= argc)
			{
				printf("[Fatal] -prio given without a value\n");
				return 1;
			}
			char *endp = 0;
			long v = strtol(argv[i + 1], &endp, 10);
			if (endp == argv[i + 1] || (endp && *endp))
			{
				printf("[Fatal] -prio value '%s' is not a number\n", argv[i + 1]);
				return 1;
			}
			worker_prio = (int)v;
			i++;
			continue;
		}
		if (strcmp(argv[i], "-maxgen") == 0)
		{
			/* server-wide default generation budget; 0 = unlimited */
			if (i + 1 >= argc)
			{
				printf("[Fatal] -maxgen given without a value\n");
				return 1;
			}
			char *endp = 0;
			long v = strtol(argv[i + 1], &endp, 10);
			if (endp == argv[i + 1] || (endp && *endp))
			{
				printf("[Fatal] -maxgen value '%s' is not a number\n", argv[i + 1]);
				return 1;
			}
			if (v < 0)
			{
				printf("[Fatal] -maxgen %ld is negative\n", v);
				return 1;
			}
			g_max_gen = (v > 2147483647L) ? 2147483647 : (int)v;
			i++;
			continue;
		}
		if (strcmp(argv[i], "-history") == 0)
		{
			/* web chat history capacity; the oldest pair is evicted when full */
			if (i + 1 >= argc)
			{
				printf("[Fatal] -history given without a value\n");
				return 1;
			}
			char *endp = 0;
			long v = strtol(argv[i + 1], &endp, 10);
			if (endp == argv[i + 1] || (endp && *endp))
			{
				printf("[Fatal] -history value '%s' is not a number\n", argv[i + 1]);
				return 1;
			}
			if (v < 2)
			{
				printf("[Fatal] -history %ld is below the minimum of 2 (a user+bot pair)\n", v);
				return 1;
			}
			server_history = (v > 2147483647L) ? 2147483647 : (int)v;
			i++;
			continue;
		}
		if (strcmp(argv[i], "-stream") == 0)
		{
			/* StreamingLLM opt-in: 1 = the ring KV cache with attention sinks
			 * replaces the legacy full-context path for every session. */
			if (i + 1 >= argc)
			{
				printf("[Fatal] -stream given without a value\n");
				return 1;
			}
			char *endp = 0;
			long v = strtol(argv[i + 1], &endp, 10);
			if (endp == argv[i + 1] || (endp && *endp))
			{
				printf("[Fatal] -stream value '%s' is not a number\n", argv[i + 1]);
				return 1;
			}
			server_stream = (v != 0);
			i++;
			continue;
		}
		if (strcmp(argv[i], "-sinks") == 0)
		{
			/* attention-sink count S: the first S tokens of the session are
			 * kept forever; only meaningful with -stream 1. The value passes
			 * through untouched: the engine validates its own domain. */
			if (i + 1 >= argc)
			{
				printf("[Fatal] -sinks given without a value\n");
				return 1;
			}
			char *endp = 0;
			long v = strtol(argv[i + 1], &endp, 10);
			if (endp == argv[i + 1] || (endp && *endp))
			{
				printf("[Fatal] -sinks value '%s' is not a number\n", argv[i + 1]);
				return 1;
			}
			server_sinks = (int)v;
			i++;
			continue;
		}
		if (strcmp(argv[i], "-window") == 0)
		{
			/* sliding window W: the newest W non-sink tokens stay resident;
			 * older tokens wrap (and surface as evictions to the sidecar);
			 * only meaningful with -stream 1. The value passes through
			 * untouched: the engine validates its own domain. */
			if (i + 1 >= argc)
			{
				printf("[Fatal] -window given without a value\n");
				return 1;
			}
			char *endp = 0;
			long v = strtol(argv[i + 1], &endp, 10);
			if (endp == argv[i + 1] || (endp && *endp))
			{
				printf("[Fatal] -window value '%s' is not a number\n", argv[i + 1]);
				return 1;
			}
			server_window = (int)v;
			i++;
			continue;
		}
		if (argv[i][0] == '-')
		{
			/* An unknown flag must never become a positional (a "-foo" DLL
			 * path fails much later and much less clearly) and must never
			 * be ignored (the operator believes a setting took effect). */
			printf("[Fatal] unknown flag '%s'\n\n", argv[i]);
			printf("Usage: %s <library> <lz4ht.db> <port> <model_id> [-prio N] [-maxgen N] [-history N]\n"
			       "                 [-stream 0|1] [-sinks N] [-window N]\n", argv[0]);
			return 1;
		}
		if (num_positionals >= 4)
		{
			printf("[Fatal] unexpected extra argument '%s' (expected exactly: <dll> <db> <port> <model_id>)\n", argv[i]);
			return 1;
		}
		positionals[num_positionals++] = argv[i];
	}
	/* Win32 SetThreadPriority valid values depend on the process priority
	 * class (MSDN SetThreadPriority): a NORMAL-class process accepts only
	 * {-15 (idle), -2..2 (below/normal/above/highest), 15 (time-critical)}.
	 * Raw values 3..7 and -3..-7 require REALTIME_PRIORITY_CLASS, which this
	 * server never sets. Values outside the valid set make SetThreadPriority
	 * fail at pool creation -- refuse HERE, before the value reaches the
	 * engine. */
	if (worker_prio != -15 && (worker_prio < -2 || worker_prio > 2) && worker_prio != 15)
	{
		printf("[Fatal] -prio %d is not a valid thread priority for a NORMAL-class process (valid: -15, -2..2, 15)\n", worker_prio);
		return 1;
	}

	if (num_positionals < 4)
	{
		printf("Usage: %s <library> <lz4ht.db> <port> <model_id> [-prio N] [-maxgen N] [-history N]\n"
		       "                 [-stream 0|1] [-sinks N] [-window N]\n"
		       "\n"
		       "  <library> is the engine library path: v7_threading.dll on Windows,\n"
		       "  libv7_threading.so on Linux.\n"
		       "\n"
		       "  -prio N     Win32 worker priority for v7 (-15 idle .. 15 time critical).\n"
	       "              Default -15 (idle): the machine stays responsive while the\n"
	       "              server serves. Applied via v7_set_worker_priority (deferred\n"
	       "              to pool creation when the pool does not exist yet) and\n"
	       "              v7_override_prio at pool creation.\n"
	       "  -maxgen N   Server-wide default generation budget in tokens. 0 =\n"
	       "              unlimited (default): a turn runs until EOS or the output\n"
	       "              buffer is full. A request's own max_tokens still wins\n"
	       "              when it is greater than 0.\n"
	       "  -history N  Web chat history entry capacity (default 4096, minimum\n"
	       "              2). When full, the oldest user+bot pair is evicted --\n"
	       "              turns never vanish silently while inference runs.\n"
	       "  -stream 0|1 StreamingLLM for every chat session (default 0 = legacy\n"
	       "              full-context KV). With 1, sessions older than the fill\n"
	       "              grade wrap instead of growing, and evicted text is\n"
	       "              delivered to the SectorLogo sidecar via on_stream_evict.\n"
	       "              The server has no INI of its own: the CLI flags are the\n"
	       "              only configuration surface (server.bat maps the\n"
	       "              V7_SERVER_STREAM / V7_SERVER_SINKS / V7_SERVER_WINDOW env\n"
	       "              vars onto them).\n"
	       "  -sinks N    StreamingLLM attention-sink count S (default 4). The first\n"
	       "              S tokens of the session are never evicted.\n"
	       "  -window N   StreamingLLM sliding window W (default 996; fill grade\n"
	       "              C = S + W). The newest W non-sink tokens stay resident.\n",
	       argv[0]);
		return 1;
	}

	/* Unbuffered stdout: under the logfile redirect (server.bat >> log 2>&1)
	 * the CRT full-buffers printf output, and the /quit exit path never
	 * flushes it -- every diagnostic line up to the last 4 KB would be lost.
	 * Unbuffered makes each printf land in the log the moment it prints,
	 * same as the console window. */
	setvbuf(stdout, 0, _IONBF, 0);
	/* UTF-8 console output: model answers are UTF-8 and a legacy codepage
	 * garbles every non-ASCII byte on its way to the window. */
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif

	/* Response ids (chatcmpl-N) must not repeat across restarts of the bat
	 * loop: seed the counter from the wall clock, not from 0. */
	request_counter.store((int)(GetTickCount64() & 0x3FFFFFFF) + 1);

	/* the web chat history array: malloc'd to the -history capacity */
	g_chat_history_cap = server_history;
	g_chat_history = (chat_msg *)malloc(sizeof(chat_msg) * (size_t)server_history);
	memset(g_chat_history, 0, sizeof(chat_msg) * (size_t)server_history);

	char *dll_path = positionals[0];
	char *db_path = positionals[1];
	int port = DEFAULT_PORT;
	{
		char *endp = 0;
		long port_val = strtol(positionals[2], &endp, 10);
		if (endp == positionals[2] || (endp && *endp))
		{
			printf("[Fatal] port '%s' is not a number\n", positionals[2]);
			return 1;
		}
		port = (int)port_val;
	}
	if (port <= 0 || port > 65535)
	{
		printf("[Fatal] port %d is outside 1..65535\n", port);
		return 1;
	}
	/* positional 4: the model id reported by /v1/models; no default -- the
	 * launcher says which model it is serving */
	const char *model_id = positionals[3];
	g_health_model_id = model_id; /* health thread serves /v1/models with the same id */

	/* ---- 1. Load DLL ---- */
	if (v7_load(dll_path, &server_dll) != 0)
	{
		printf("[Fatal] failed to load %s\n", dll_path);
		return 1;
	}
	printf("[v7_server] engine version %s\n", server_dll.v7_get_version());

	/* ---- 2. Initialize database connection ---- */
	printf("[v7_server] loading %s...\n", db_path);
	if (server_dll.v7_init((char *)db_path) != 1)
	{
		if (server_dll.v7_engine_is_dead())
			printf("[Fatal] V7_init(\"%s\") died: %s\n", db_path, server_dll.v7_get_death_reason());
		else
			printf("[Fatal] V7_init(\"%s\") failed\n", db_path);
		v7_unload(&server_dll);
		return 1;
	}

	/* ---- 2b. Initialize COM (multithreaded: this process makes no STA
	 * COM calls and MTA needs no message pump), then GDI+ for vision image
	 * decoding. GDI+/WIC codecs and IStream depend on COM. ---- */
#ifdef _WIN32
	com_hr = CoInitializeEx(0, COINIT_MULTITHREADED);
	Gdiplus::GdiplusStartupInput gdiplus_si;
	Gdiplus::Status gdiplus_status = Gdiplus::GdiplusStartup(&g_gdiplus_token, &gdiplus_si, 0);
	if (gdiplus_status != Gdiplus::Ok)
	{
		/* GDI+ is not optional: every vision request would die later with a
		 * misleading "could not be decoded" 400. Fail NOW, visibly. */
		printf("[Fatal] GdiplusStartup failed: %d\n", (int)gdiplus_status);
		g_gdiplus_token = 0;
		if (SUCCEEDED(com_hr))
			CoUninitialize();
		v7_unload(&server_dll);
		return 1;
	}

#endif

	/* ---- 2c. Worker priority: default idle so the PC keeps working while the
	 * server serves. The engine defers a setter call that arrives before the
	 * worker pool exists (the pool is created lazily by the first dispatched
	 * op) and applies it at pool creation; the env var is the launch-time
	 * alternative for hosts that cannot call the setter. ---- */
	char prio_env[16];
	snprintf(prio_env, sizeof(prio_env), "%d", worker_prio);
	SetEnvironmentVariableA("v7_override_prio", prio_env);

	/* ---- 3. Initialize inference from microdb (context hidden in BSS) ---- */
	printf("[v7_server] initializing model...\n");
	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real return
	 * AND the death flag must both be checked. */
	if ((!server_dll.v7_initialize_inference()) || server_dll.v7_engine_is_dead())
	{
		if (server_dll.v7_engine_is_dead())
			printf("[Fatal] inference init died: %s\n", server_dll.v7_get_death_reason());
		else
			printf("[Fatal] v7_initialize_inference failed\n");
		server_platform_cleanup();
		v7_unload(&server_dll);
		return 1;
	}
	server_dll.v7_set_quantize(1); /* Q8 quantized activation mode (void setter: the engine fixes the mode, it cannot fail) */
	server_dll.v7_set_worker_priority(worker_prio); /* applied now, or deferred to pool creation when the pool does not exist yet */

	/* ---- StreamingLLM opt-in: applies to EVERY chat session this server
	 * serves (the engine keeps the flag across turns; each turn just runs
	 * longer and wraps at the fill grade C = S + W). A refusal means the
	 * flag combination cannot be served (an M-RoPE model cannot stream) --
	 * the operator asked for a mode the model cannot run, so exit instead
	 * of silently serving a different KV path. Both the return code AND
	 * the death flag are checked (gated entries). */
	if (server_stream)
	{
		int stream_rc = server_dll.v7_set_streaming(1, server_sinks, server_window);
		if (stream_rc == 0)
			printf("[v7_server] StreamingLLM enabled (sinks %d, window %d, fill grade C = %d)\n",
			       server_sinks, server_window, server_sinks + server_window);
		else if (server_dll.v7_engine_is_dead())
		{
			printf("[Fatal] v7_set_streaming died: %s\n", server_dll.v7_get_death_reason());
			server_platform_cleanup();
			server_dll.v7_free_inference();
			v7_unload(&server_dll);
			return 1;
		}
		else
		{
			printf("[Fatal] -stream 1 refused by the engine (rc %d: an M-RoPE model cannot stream, or S/W out of range)\n", stream_rc);
			server_platform_cleanup();
			server_dll.v7_free_inference();
			v7_unload(&server_dll);
			return 1;
		}
	}

	/* v7_set_sampler_topk is NOT called on purpose: the engine's sampler
	 * (v7s_sample) implements temperature + top_p only and FATALs on any
	 * top_k != 0 (observed: v7_set_sampler_topk(20) killed the engine at
	 * load). In the Qwen3 filter chain top_k precedes top_p, and the top_p
	 * 0.95 cut is the wider of the two, so the temp 0.6 / top_p 0.95 pair
	 * carries the model-card recommendation without it. */
	set_engine_thinking(1); /* thinking ON: also arms the Qwen3 thinking pair (temp 0.6 / top_p 0.95); model generates <think> block naturally */
	/* Debug aid: V7_SERVER_TOKDUMP=1 turns on the DLL's exported token-stream
	 * dump (token ids + KV positions for every prefill/generate step) so the
	 * server log shows exactly what context each turn saw. */
	char tokdump_env[8];
	DWORD tokdump_need = GetEnvironmentVariableA("V7_SERVER_TOKDUMP", tokdump_env, sizeof(tokdump_env));
	if (tokdump_need > 0 && tokdump_need < sizeof(tokdump_env) && tokdump_env[0] == '1')
		*server_dll.enable_token_dump = 1;
	printf("[v7_server] model loaded (worker priority %d)\n", worker_prio);
	/* ---- 4. Winsock2 init ---- */
#ifdef _WIN32
	WSADATA wsa;
	int wsa_err = WSAStartup(MAKEWORD(2, 2), &wsa);
	if (wsa_err != 0)
	{
		printf("[Fatal] WSAStartup failed: %d\n", wsa_err);
		server_platform_cleanup();
		server_dll.v7_free_inference();
		v7_unload(&server_dll);
		return 1;
	}

#endif

	/* ---- 4. Create listening socket ---- */
#ifdef __linux__
	SOCKET listen_fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, IPPROTO_TCP);
#else
	SOCKET listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#endif
	if (listen_fd == INVALID_SOCKET)
	{
		printf("[Fatal] socket() failed: %d\n", WSAGetLastError());
		server_network_cleanup();
		server_platform_cleanup();
		server_dll.v7_free_inference();
		v7_unload(&server_dll);
		return 1;
	}

	/* SO_EXCLUSIVEADDRUSE (NOT SO_REUSEADDR): on Windows, SO_REUSEADDR lets
	 * ANY other process bind the same port, and a more specific bind (e.g.
	 * an attacker's 127.0.0.1:8080) wins over our wildcard — a socket
	 * hijack vector. Exclusive-use gains the restart convenience nothing
	 * here (the listening socket is closed before the loop restarts). */
	int optval = 1;
#ifdef __linux__
	setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
#else
	setsockopt(listen_fd, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (char *)&optval, sizeof(optval));
#endif

	/* Bind address: 127.0.0.1 by DEFAULT (this is a local application -- the
	 * operator opts into a network bind explicitly). A return value >= the
	 * buffer size means the value did NOT fit -- using a truncated value
	 * would inet_pton garbage, so the long value is refused with the same
	 * message as an invalid one. */
	const char *bind_addr = "127.0.0.1";
	char bind_env[64];
	DWORD bind_need = GetEnvironmentVariableA("V7_SERVER_BIND", bind_env, sizeof(bind_env));
	if (bind_need > 0)
	{
		if (bind_need >= sizeof(bind_env) || !bind_env[0])
		{
			printf("[Fatal] V7_SERVER_BIND is empty or too long (max %llu chars)\n", (unsigned long long)sizeof(bind_env) - 1);
			closesocket(listen_fd);
			server_network_cleanup();
			server_platform_cleanup();
			server_dll.v7_free_inference();
			v7_unload(&server_dll);
			return 1;
		}
		bind_addr = bind_env;
	}

	struct sockaddr_in addr;
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons((u_short)port);
	if (inet_pton(AF_INET, bind_addr, &addr.sin_addr) != 1)
	{
		printf("[Fatal] V7_SERVER_BIND=\"%s\" is not a valid IPv4 address\n", bind_addr);
		closesocket(listen_fd);
		server_network_cleanup();
		server_platform_cleanup();
		server_dll.v7_free_inference();
		v7_unload(&server_dll);
		return 1;
	}

	/* Bind once: whoever holds the port owns it. A busy port means another
	 * server is already running -- report the error and terminate with the
	 * port-busy code so the supervisor stops instead of relaunching into the
	 * same conflict. */
	if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
	{
		int bind_err = WSAGetLastError();
		printf("[Fatal] bind(%s:%d) failed: %d -- port %d is already in use\n", bind_addr, port, bind_err, port);
		closesocket(listen_fd);
		server_network_cleanup();
		server_platform_cleanup();
		server_dll.v7_free_inference();
		v7_unload(&server_dll);
		return SERVER_EXIT_PORT_BUSY;
	}

	if (listen(listen_fd, 16) != 0)
	{ /* 16: browsers open several conns at once */
		printf("[Fatal] listen() failed: %d\n", WSAGetLastError());
		closesocket(listen_fd);
		server_network_cleanup();
		server_platform_cleanup();
		server_dll.v7_free_inference();
		v7_unload(&server_dll);
		return 1;
	}

	/* ---- 5b. Disable console QuickEdit mode ----
	 * QuickEdit (a click/selection inside a cmd console) SUSPENDS all console
	 * writes process-wide until the selection is cleared. Our printf of full
	 * model responses would then block the single server thread forever at
	 * 0% CPU — the observed unkillable hang. */
#ifdef _WIN32
	HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
	DWORD cm;
	if (GetConsoleMode(hIn, &cm))
	{
		cm &= ~ENABLE_QUICK_EDIT_MODE;
		cm |= ENABLE_EXTENDED_FLAGS; /* required for QuickEdit changes */
		SetConsoleMode(hIn, cm);
	}

#endif

	printf("[v7_server] listening on http://%s:%d\n", bind_addr, port);

	/* ---- 5c. Health thread: the SINGLE acceptor. /v1/health + /v1/models
	 * stay ALIVE during long inferences (watchdogs must not kill a healthy
	 * server); it peeks at the request line and hands every OTHER
	 * connection to the main thread through the handoff queue. TWO
	 * acceptors on one blocking socket RACE: when both threads sit in
	 * accept(), one connection wakes exactly one of them -- a connection
	 * the health thread wins lands in the handoff queue while the main
	 * thread stays BLOCKED in its own accept() with no second connection
	 * in sight. One acceptor + a handoff event has no such window. The
	 * handle is kept so cleanup can JOIN the thread BEFORE Winsock is torn
	 * down (a mid-send health answer racing WSACleanup is undefined
	 * behavior). ---- */
	InitializeCriticalSection(&g_handoff_lock);
#ifdef __linux__
	g_handoff_event = server_create_event();
#else
	g_handoff_event = CreateEvent(0, FALSE, FALSE, 0); /* auto-reset: one signal per queued connection */
#endif
	if (!g_handoff_event)
	{
		printf("[Fatal] CreateEvent failed: %lu\n", GetLastError());
		closesocket(listen_fd);
		server_network_cleanup();
		server_platform_cleanup();
		server_dll.v7_free_inference();
		v7_unload(&server_dll);
		return 1;
	}
#ifdef __linux__
	pthread_t health_thread_handle;
	int thread_error = pthread_create(&health_thread_handle, nullptr, linux_health_thread, (void *)(uintptr_t)listen_fd);
	if (thread_error)
		errno = thread_error;
	if (thread_error)
#else
	HANDLE health_thread_handle = CreateThread(0, 0, health_thread, (LPVOID)(uintptr_t)listen_fd, 0, 0);
	if (!health_thread_handle)
#endif
	{
		/* the health thread is the ONLY acceptor: without it the server
		 * listens but never answers -- fail visibly instead of going deaf */
		printf("[Fatal] CreateThread for the health acceptor failed: %lu\n", GetLastError());
#ifdef __linux__
		server_close_event(g_handoff_event);
#else
		CloseHandle(g_handoff_event);
#endif
		DeleteCriticalSection(&g_handoff_lock);
		closesocket(listen_fd);
		server_network_cleanup();
		server_platform_cleanup();
		server_dll.v7_free_inference();
		v7_unload(&server_dll);
		return 1;
	}

	/* ---- 6. Serve loop ---- */
	while (server_running.load())
	{
		/* Wait for a handed-off connection (event) or the 1 s timeout (the
		 * server_running / shutdown re-check). The main thread never calls
		 * accept() -- the health thread is the single acceptor. */
#ifdef __linux__
		server_wait_event(g_handoff_event, 1000);
#else
		WaitForSingleObject(g_handoff_event, 1000);
#endif
		for (;;)
		{
			if (!server_running.load())
				break; /* /quit or engine death: stop serving mid-drain, the cleanup below closes the leftovers */
			SOCKET queued = handoff_pop();
			if (queued == INVALID_SOCKET)
				break;
			serve_client(queued, (char *)model_id); /* model_id is the required argv id: served read-only */
		}
	}

	/* ---- 7. Cleanup ---- */
#ifdef __linux__
	shutdown(listen_fd, SHUT_RDWR); /* Wake the acceptor before joining it. */
	pthread_join(health_thread_handle, nullptr);
	closesocket(listen_fd);
	for (SOCKET queued; (queued = handoff_pop()) != INVALID_SOCKET;)
		closesocket(queued);
	DeleteCriticalSection(&g_handoff_lock);
	server_close_event(g_handoff_event);
#else
	closesocket(listen_fd); /* accept fails -> the health thread exits its loop */
	if (health_thread_handle)
	{
		/* The health thread's per-recv cap is 10 s; a thread parked inside a
		 * recv cannot observe server_running until that recv returns. Wait
		 * LONGER than the socket timeout, then tear Winsock down. But only
		 * tear down what the thread still SHARES after a confirmed exit:
		 * a stuck thread (join timeout) can still call handoff_push, and
		 * deleting the critical section or closing the event it uses is
		 * undefined behavior -- both stay alive for the OS to reclaim at
		 * process exit. */
		DWORD join_result = WaitForSingleObject(health_thread_handle, 12000);
		if (join_result == WAIT_OBJECT_0)
		{
			CloseHandle(health_thread_handle);
			for (;;)
			{
				SOCKET queued = handoff_pop();
				if (queued == INVALID_SOCKET)
					break;
				closesocket(queued); /* connections still queued at shutdown: never leak them */
			}
			DeleteCriticalSection(&g_handoff_lock);
			if (g_handoff_event)
			{
				CloseHandle(g_handoff_event);
				g_handoff_event = 0;
			}
		}
		else
		{
			printf("[v7_server] health thread did not exit within 12 s -- leaving its handoff lock/event alive\n");
		}
	}
#endif
	server_dll.v7_free_inference();
	server_dll.v7_shutdown();
	v7_unload(&server_dll);
	server_platform_cleanup();
	server_network_cleanup();
	return server_exit_code.load(); /* /quit?code=N (default 42) -- the bat loop stops on 42 */
}
