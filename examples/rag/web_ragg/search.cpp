// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#include "search.h"
#include "http_client.h"
#include "text_utils.h"
#include "html_parser.h"
#include <algorithm>

std::vector<search_result> searchWeb(std::string &query, int maxResults)
{
	std::string url = "https://lite.duckduckgo.com/lite/?q=" + urlEncode(query);

	std::string html = curlGetBody(url, 15);
	if (html.empty())
	{
		return std::vector<search_result>();
	}

	return parseSearchResults(html, maxResults);
}

std::vector<search_result> parseSearchResults(std::string &html, int maxResults)
{
	std::vector<search_result> results;

	int pos = 0;
	int count = 0;

	while (count < maxResults)
	{
		int linkPos = html.find("class='result-link'", pos);
		if (linkPos == std::string::npos)
			break;

		int hrefStart = html.rfind("href=\"", linkPos);
		if (hrefStart != std::string::npos && linkPos - hrefStart > 200)
			hrefStart = std::string::npos;
		int hrefAlt = html.rfind("href='", linkPos);
		if (hrefAlt != std::string::npos && linkPos - hrefAlt > 200)
			hrefAlt = std::string::npos;
		if (hrefStart == std::string::npos)
			hrefStart = hrefAlt;
		else if (hrefAlt != std::string::npos && hrefAlt > hrefStart)
			hrefStart = hrefAlt;

		if (hrefStart == std::string::npos)
		{
			pos = linkPos + 1;
			continue;
		}

		char delim = html[hrefStart + 5];
		int hrefEnd = html.find(delim, hrefStart + 6);
		if (hrefEnd == std::string::npos || hrefEnd > linkPos + 150)
		{
			pos = linkPos + 1;
			continue;
		}

		std::string rawUrl = html.substr(hrefStart + 6, hrefEnd - hrefStart - 6);

		int closeTag = html.find("</a>", linkPos);
		if (closeTag == std::string::npos)
		{
			pos = linkPos + 1;
			continue;
		}

		int textStart = html.find(">", linkPos);
		if (textStart == std::string::npos || textStart >= closeTag)
		{
			pos = closeTag + 1;
			continue;
		}
		std::string title = html.substr(textStart + 1, closeTag - textStart - 1);
		title = cleanText(title);

		std::string snippet;
		int snippetPos = html.find("class='result-snippet'", closeTag);
		if (snippetPos != std::string::npos)
		{
			int tdStart = html.find(">", snippetPos);
			int tdEnd = html.find("</td>", tdStart);
			if (tdStart != std::string::npos && tdEnd != std::string::npos)
			{
				snippet = html.substr(tdStart + 1, tdEnd - tdStart - 1);
				snippet = cleanText(snippet);
			}
		}

		std::string linkUrl;
		int uddgPos = rawUrl.find("uddg=");
		if (uddgPos != std::string::npos)
		{
			linkUrl = rawUrl.substr(uddgPos + 5);

			int ampPos = 0;
			while ((ampPos = linkUrl.find("&amp;", ampPos)) != std::string::npos)
			{
				linkUrl.replace(ampPos, 5, "&");
			}

			linkUrl = urlDecode(linkUrl);

			int rutPos = linkUrl.find("&rut=");
			if (rutPos != std::string::npos)
			{
				linkUrl = linkUrl.substr(0, rutPos);
			}
		}
		else
		{
			linkUrl = urlDecode(rawUrl);
		}

		if (linkUrl.find("http") != 0)
		{
			pos = closeTag + 1;
			continue;
		}

		if (!title.empty() && !linkUrl.empty())
		{
			search_result r;
			r.title = title;
			r.snippet = snippet;
			r.url = linkUrl;
			results.push_back(r);
			count++;
		}

		pos = closeTag + 1;
	}

	return results;
}