// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

#include <string>
#include <vector>

std::string decodeEntities(std::string &text);
std::string stripTags(std::string &html);
std::string normalizeWhitespace(std::string &text);
std::string cleanText(std::string &html);
std::string urlDecode(std::string &encoded);
std::string urlEncode(std::string &text);