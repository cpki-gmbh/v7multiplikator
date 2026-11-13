// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

#include <string>

enum shortener_mode
{
	SHORTENER_SIMPLE,
	SHORTENER_LLM
};

void shortener_set_mode(shortener_mode mode);
shortener_mode shortener_get_mode();
int shortener_init(char *dll_path, char *ini_path);
void shortener_shutdown();
std::string shortenText(std::string &content, int maxChars);
std::string shortenTextSmart(std::string &content, int maxChars);
std::string shortenTextQuery(std::string &content, int maxChars, std::string &query);
