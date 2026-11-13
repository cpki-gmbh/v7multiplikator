// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

#include <string>
#include <vector>
#include "search.h"

typedef struct page_link
{
	int number;
	std::string text;
	std::string url;
} page_link;

typedef struct page_info
{
	std::string title;
	std::string url;
	std::string snippet;
	std::string content;
	std::vector<page_link> links;
} page_info;

page_info parsePage(std::string &html, std::string &url);
std::vector<search_result> search(std::string &query, int maxResults = 10);
std::string fetchUrl(std::string &url, int timeoutSec = 30);
int fetchUrlStatus(std::string &url, int timeoutSec = 30);
std::string getDebugDir();
void setDebugDir(std::string &dir);