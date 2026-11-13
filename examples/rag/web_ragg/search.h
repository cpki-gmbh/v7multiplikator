// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

#include <string>
#include <vector>

typedef struct search_result
{
	std::string title;
	std::string snippet;
	std::string url;
} search_result;

std::vector<search_result> searchWeb(std::string &query, int maxResults = 10);
std::vector<search_result> parseSearchResults(std::string &html, int maxResults = 10);