// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#include "text_utils.h"
#include <algorithm>
#include <cctype>
#include <cstdio>

char *ENTITIES[][2] = { { "&nbsp;", " " },     { "&amp;", "&" },     { "&lt;", "<" },      { "&gt;", ">" },      { "&quot;", "\"" },  { "&#39;", "'" },    { "&apos;", "'" },   { "&mdash;", "-" },  { "&ndash;", "-" },
	                    { "&hellip;", "..." }, { "&deg;", " deg" },  { "&euro;", "EUR" },  { "&pound;", "GBP" }, { "&auml;", "ae" },  { "&ouml;", "oe" },  { "&uuml;", "ue" },  { "&szlig;", "ss" }, { "&eacute;", "e" },
	                    { "&egrave;", "e" },   { "&ecirc;", "e" },   { "&agrave;", "a" },  { "&acirc;", "a" },   { "&igrave;", "i" }, { "&ograve;", "o" }, { "&ocirc;", "o" },  { "&ugrave;", "u" }, { "&ucirc;", "u" },
	                    { "&lsquo;", "'" },    { "&rsquo;", "'" },   { "&ldquo;", "\"" },  { "&rdquo;", "\"" },  { "&#8217;", "'" },  { "&#8216;", "'" },  { "&#8220;", "\"" }, { "&#8221;", "\"" }, { "&#x2019;", "'" },
	                    { "&#x2018;", "'" },   { "&#x201C;", "\"" }, { "&#x201D;", "\"" }, { "&#91;", "[" },     { "&#93;", "]" },    { "&#160;", " " },   { "&#8230;", "..." } };

int NUM_ENTITIES = sizeof(ENTITIES) / sizeof(ENTITIES[0]);

std::string decodeEntities(std::string &text)
{
	std::string result = text;

	for (int i = 0; i < NUM_ENTITIES; i++)
	{
		int pos = 0;
		while ((pos = result.find(ENTITIES[i][0], pos)) != std::string::npos)
		{
			result.replace(pos, strlen(ENTITIES[i][0]), ENTITIES[i][1]);
			pos += strlen(ENTITIES[i][1]);
		}
	}

	int pos = 0;
	while ((pos = result.find("&", pos)) != std::string::npos)
	{
		int semi = result.find(";", pos);
		if (semi != std::string::npos && semi - pos < 10)
		{
			int nested = result.find("&", pos + 1);
			if (nested == std::string::npos || nested > semi)
			{
				result.erase(pos, semi - pos + 1);
			}
			else
			{
				pos = nested;
			}
		}
		else
		{
			pos++;
		}
	}

	return result;
}

std::string stripTags(std::string &html)
{
	std::string result = html;
	int pos = 0;

	while ((pos = result.find("<", pos)) != std::string::npos)
	{
		int end = result.find(">", pos);
		if (end == std::string::npos)
			break;
		if (end > pos)
		{
			result.erase(pos, end - pos + 1);
		}
		else
		{
			pos++;
		}
	}

	return result;
}

std::string normalizeWhitespace(std::string &text)
{
	std::string result;
	bool lastSpace = true;

	for (int i = 0; i < text.size(); i++)
	{
		char c = text[i];
		if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
		{
			if (!lastSpace)
			{
				result += ' ';
				lastSpace = true;
			}
		}
		else
		{
			result += c;
			lastSpace = false;
		}
	}

	while (!result.empty() && result[0] == ' ')
		result.erase(result.begin());
	while (!result.empty() && result.back() == ' ')
		result.pop_back();

	return result;
}

std::string cleanText(std::string &html)
{
	std::string text = stripTags(html);
	text = decodeEntities(text);
	text = normalizeWhitespace(text);
	return text;
}

std::string urlDecode(std::string &encoded)
{
	std::string result;
	int i = 0;

	while (i < encoded.size())
	{
		if (encoded[i] == '%' && i + 2 < encoded.size())
		{
			char h1 = encoded[i + 1];
			char h2 = encoded[i + 2];
			bool valid1 = (h1 >= '0' && h1 <= '9') || (h1 >= 'A' && h1 <= 'F') || (h1 >= 'a' && h1 <= 'f');
			bool valid2 = (h2 >= '0' && h2 <= '9') || (h2 >= 'A' && h2 <= 'F') || (h2 >= 'a' && h2 <= 'f');
			if (valid1 && valid2)
			{
				int val = 0;
				if (h1 >= '0' && h1 <= '9')
					val = h1 - '0';
				else if (h1 >= 'A' && h1 <= 'F')
					val = h1 - 'A' + 10;
				else
					val = h1 - 'a' + 10;
				val <<= 4;
				if (h2 >= '0' && h2 <= '9')
					val += h2 - '0';
				else if (h2 >= 'A' && h2 <= 'F')
					val += h2 - 'A' + 10;
				else
					val += h2 - 'a' + 10;
				result += (char)val;
				i += 3;
			}
			else
			{
				result += '%';
				i++;
			}
		}
		else if (encoded[i] == '+')
		{
			result += ' ';
			i++;
		}
		else
		{
			result += encoded[i];
			i++;
		}
	}

	return result;
}

std::string urlEncode(std::string &text)
{
	std::string result;

	for (int i = 0; i < text.size(); i++)
	{
		char c = text[i];
		if (std::isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~')
		{
			result += c;
		}
		else if (c == ' ')
		{
			result += "+";
		}
		else
		{
			char buf[8];
			snprintf(buf, sizeof(buf), "%%%02X", (unsigned char)c);
			result += buf;
		}
	}

	return result;
}