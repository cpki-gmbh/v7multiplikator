// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#include "webtool.h"
#include "html_parser.h"
#include "text_utils.h"
#include "http_client.h"
#include "search.h"
#include <fstream>
#include <algorithm>
#include <windows.h>

std::string web_debug_dump_dir = "C:\\webagent\\build\\";

bool isFileUrl(std::string &url)
{
	return url.size() > 8 && url.substr(0, 8) == "file:///";
}

std::string urlToFilePath(std::string &url)
{
	std::string path = url.substr(8);
	if (path.size() >= 2 && path[1] == ':')
	{
		path[0] = toupper(path[0]);
	}
	std::replace(path.begin(), path.end(), '/', '\\');
	return path;
}

std::string readFileContent(std::string &path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		return "";
	std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	file.close();
	return content;
}

void setDebugDir(std::string &dir)
{
	web_debug_dump_dir = dir;
	if (!web_debug_dump_dir.empty() && web_debug_dump_dir[web_debug_dump_dir.size() - 1] != '\\' && web_debug_dump_dir[web_debug_dump_dir.size() - 1] != '/')
	{
		web_debug_dump_dir += "\\";
	}
}

std::string getDebugDir()
{
	return web_debug_dump_dir;
}

std::string extractDomain(std::string &url)
{
	int pos = url.find("://");
	if (pos == std::string::npos)
		return url;
	pos += 3;
	int end = url.find("/", pos);
	if (end == std::string::npos)
	{
		return url.substr(pos);
	}
	std::string domain = url.substr(pos, end - pos);
	return domain;
}

page_info parsePage(std::string &html, std::string &url)
{
	page_info info;
	info.url = url;

	info.title = extractTitle(html);

	std::string og_name = "og:description";
	std::string desc_name = "description";
	std::string ogDesc = extractMetaContent(html, og_name);
	std::string metaDesc = extractMetaContent(html, desc_name);
	if (!metaDesc.empty())
		info.snippet = metaDesc;
	else if (!ogDesc.empty())
		info.snippet = ogDesc;

	std::string content = findContentArea(html);

	std::vector<std::string> blockTags = { "script", "style", "noscript", "nav", "header", "footer", "svg" };
	content = removeBlocks(content, blockTags);

	std::vector<std::string> sidebarClasses = { "sidebar", "related", "comments", "menu", "breadcrumb", "navigation", "navbar", "topbar", "cookie", "popup", "modal", "banner", "ad-", "advertisement" };
	content = removeByClass(content, sidebarClasses);

	info.content = cleanText(content);

	std::vector<parsed_link> parsedLinks = html_parser::extractLinks(content);
	for (int i = 0; i < parsedLinks.size(); i++)
	{
		page_link link;
		link.number = (int)i + 1;
		link.text = parsedLinks[i].text;
		link.url = parsedLinks[i].url;
		info.links.push_back(link);
	}

	return info;
}

std::string fetchUrl(std::string &url, int timeoutSec)
{
	if (isFileUrl(url))
	{
		std::string path = urlToFilePath(url);
		return readFileContent(path);
	}

	std::string html = curlGetBody(url, timeoutSec);

	if (!html.empty())
	{
		std::string domain = extractDomain(url);
		for (int i = 0; i < domain.size(); i++)
		{
			if (domain[i] == ':')
				domain[i] = '_';
		}
		std::string filename = web_debug_dump_dir + domain + "_page.html";

		std::ofstream out(filename.c_str(), std::ios::binary);
		if (out)
		{
			out << html;
			out.close();
		}
	}

	return html;
}

int fetchUrlStatus(std::string &url, int timeoutSec)
{
	if (isFileUrl(url))
	{
		std::string path = urlToFilePath(url);
		WIN32_FIND_DATAA fd;
		HANDLE h = FindFirstFileA(path.c_str(), &fd);
		if (h != INVALID_HANDLE_VALUE)
		{
			FindClose(h);
			return 200;
		}
		return 404;
	}
	return curlGetStatusCode(url, timeoutSec);
}

std::vector<search_result> search(std::string &query, int maxResults)
{
	return searchWeb(query, maxResults);
}