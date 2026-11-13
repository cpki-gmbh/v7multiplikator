// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

#include <string>
#include <vector>

typedef struct html_node
{
	int start;
	int end;
	std::string name;
	std::string attributes;
	bool selfContained;
} html_node;

typedef struct html_region
{
	int start;
	int end;
	int depth;
	std::string tagName;
} html_region;

typedef struct parsed_link
{
	int number;
	std::string text;
	std::string url;
	int position;
} parsed_link;

class html_parser
{
  public:
	static std::vector<html_region> findAllRegions(std::string &html);
	static html_region findRegionAt(std::string &html, int position);
	static std::string extractRegionContent(std::string &html, html_region &region);
	static std::vector<parsed_link> extractLinks(std::string &html);
	static std::string getTagName(std::string &html, int start);
	static std::string getAttribute(std::string &html, int tagStart, std::string &attrName);
	static bool isSelfContained(std::string &html, int tagStart);
};

std::string removeBlocks(std::string &html, std::vector<std::string> &blockTags);
std::string removeByClass(std::string &html, std::vector<std::string> &classNames);
std::string findContentArea(std::string &html);
std::string extractTitle(std::string &html);
std::string extractMetaContent(std::string &html, std::string &name);