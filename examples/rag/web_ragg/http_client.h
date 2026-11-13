// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

#include <string>

typedef struct http_response
{
	int statusCode;
	std::string body;
	std::string error;
	std::string finalUrl;
} http_response;

http_response curlGet(std::string &url, int timeoutSec = 30);
std::string curlGetBody(std::string &url, int timeoutSec = 30);
int curlGetStatusCode(std::string &url, int timeoutSec = 30);