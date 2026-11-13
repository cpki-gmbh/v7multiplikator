// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v7_server_json.cpp -- the sole translation unit of the JSON parser and
 * response builders. v7_server_json.h DEFINES (not declares) these
 * functions, so exactly one .cpp may include it; every other server
 * translation unit sees only the struct declarations
 * (v7_server_json_types.h) and the function declarations
 * (v7_server_common.h, extern "C").
 */
#ifdef __linux__
#include <windows.h> // Linux tool adapters.
#endif
#include "v7_server_json.h"
