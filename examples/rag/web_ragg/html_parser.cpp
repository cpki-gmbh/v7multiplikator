// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#include "html_parser.h"
#include <algorithm>
#include <cctype>
#include <cstring>

std::string toLower(std::string &s)
{
	std::string r = s;
	std::transform(r.begin(), r.end(), r.begin(), ::tolower);
	return r;
}

int findTagEnd(std::string &html, int tagStart)
{
	if (tagStart >= html.size() || html[tagStart] != '<')
		return std::string::npos;

	int i = tagStart + 1;
	if (i < html.size() && (html[i] == '!' || html[i] == '?'))
	{
		return html.find(">", tagStart);
	}

	if (i < html.size() && html[i] == '/')
	{
		return html.find(">", tagStart);
	}

	while (i < html.size())
	{
		if (html[i] == '>' || html[i] == '<')
			break;
		i++;
	}
	return html.find(">", tagStart);
}

bool isClosingTag(std::string &html, int tagStart)
{
	if (tagStart + 1 >= html.size())
		return false;
	return html[tagStart + 1] == '/';
}

bool isSelfContainedTag(std::string &html, int tagStart)
{
	int tagEnd = findTagEnd(html, tagStart);
	if (tagEnd == std::string::npos || tagEnd < 2)
		return false;

	if (tagEnd > tagStart && html[tagEnd - 1] == '/')
		return true;

	char *selfContained[] = { "br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "param", "source", "track", "wbr" };

	std::string name = html_parser::getTagName(html, tagStart);
	for (int i = 0; i < 14; i++)
	{
		if (name == selfContained[i])
			return true;
	}
	return false;
}

std::string html_parser::getTagName(std::string &html, int start)
{
	if (start >= html.size() || html[start] != '<')
		return "";

	int i = start + 1;
	if (i < html.size() && (html[i] == '!' || html[i] == '?' || html[i] == '/'))
	{
		i++;
	}

	while (i < html.size() && std::isspace((unsigned char)html[i]))
		i++;

	int nameStart = i;
	while (i < html.size() && !std::isspace((unsigned char)html[i]) && html[i] != '>' && html[i] != '/')
	{
		i++;
	}

	if (nameStart >= i)
		return "";
	std::string name = html.substr(nameStart, i - nameStart);
	std::transform(name.begin(), name.end(), name.begin(), ::tolower);
	return name;
}

bool html_parser::isSelfContained(std::string &html, int tagStart)
{
	return isSelfContainedTag(html, tagStart);
}

std::string html_parser::getAttribute(std::string &html, int tagStart, std::string &attrName)
{
	int tagEnd = findTagEnd(html, tagStart);
	if (tagEnd == std::string::npos)
		return "";

	std::string tagContent = html.substr(tagStart, tagEnd - tagStart);
	std::string search = toLower(attrName);

	std::string lower = toLower(tagContent);
	int pos = lower.find(search + "=");
	if (pos == std::string::npos)
		return "";
	// attribute name must be preceded by whitespace or be at the start of the tag
	if (pos > 0 && !isspace((unsigned char)lower[pos - 1]))
		return "";

	pos += search.length() + 1;
	while (pos < tagContent.size() && std::isspace((unsigned char)tagContent[pos]))
		pos++;

	if (pos >= tagContent.size())
		return "";

	char quote = 0;
	if (tagContent[pos] == '"' || tagContent[pos] == '\'')
	{
		quote = tagContent[pos];
		pos++;
	}

	int end = pos;
	if (quote)
	{
		end = tagContent.find(quote, pos);
		if (end == std::string::npos)
			return "";
	}
	else
	{
		while (end < tagContent.size() && !std::isspace((unsigned char)tagContent[end]) && tagContent[end] != '>')
		{
			end++;
		}
	}

	return tagContent.substr(pos, end - pos);
}

std::vector<html_region> html_parser::findAllRegions(std::string &html)
{
	std::vector<html_region> regions;
	if (html.empty())
		return regions;

	int depth = 0;
	int i = 0;
	html_region current;
	bool inComment = false;

	while (i < html.size())
	{
		if (html[i] == '<' && !inComment)
		{
			if (i + 3 < html.size() && html.substr(i, 4) == "<!--")
			{
				inComment = true;
				i += 4;
				continue;
			}

			int tagEnd = findTagEnd(html, i);
			if (tagEnd == std::string::npos)
			{
				i++;
				continue;
			}

			if (isClosingTag(html, i))
			{
				if (depth > 0)
					depth--;
				if (depth == 0 && !current.tagName.empty())
				{
					current.end = tagEnd + 1;
					regions.push_back(current);
					current = html_region();
				}
			}
			else if (isSelfContainedTag(html, i))
			{
				// self-contained: doesn't change depth
			}
			else
			{
				if (depth == 0)
				{
					current = html_region();
					current.start = i;
					current.tagName = html_parser::getTagName(html, i);
					current.depth = 1;
				}
				depth++;
			}
		}
		else if (inComment)
		{
			if (i + 2 < html.size() && html[i] == '-' && html[i + 1] == '-' && html[i + 2] == '>')
			{
				inComment = false;
				i += 3;
				continue;
			}
		}

		i++;
	}

	return regions;
}

html_region html_parser::findRegionAt(std::string &html, int position)
{
	html_region region;
	region.start = 0;
	region.end = 0;
	region.depth = 0;

	if (position >= html.size())
		return region;

	int i = position;
	while (i > 0 && html[i] != '<')
		i--;
	/* the loop exits at i == 0 or on '<'; a '<' at index 0 is valid too */
	if (html[i] != '<')
		return region;

	int tagEnd = findTagEnd(html, i);
	if (tagEnd == std::string::npos)
		return region;

	if (isClosingTag(html, i))
		return region;

	region.start = i;
	region.tagName = html_parser::getTagName(html, i);
	region.depth = 1;

	if (isSelfContainedTag(html, i))
	{
		region.end = tagEnd + 1;
		return region;
	}

	int currentDepth = 1;
	int scan = tagEnd + 1;

	while (scan < html.size() && currentDepth > 0)
	{
		if (html[scan] == '<')
		{
			if (scan + 3 < html.size() && html.substr(scan, 4) == "<!--")
			{
				int commentEnd = html.find("-->", scan + 4);
				if (commentEnd == std::string::npos)
					break;
				scan = commentEnd + 3;
				continue;
			}

			int nextTagEnd = findTagEnd(html, scan);
			if (nextTagEnd == std::string::npos)
				break;

			if (isClosingTag(html, scan))
			{
				currentDepth--;
			}
			else if (!isSelfContainedTag(html, scan))
			{
				currentDepth++;
			}

			scan = nextTagEnd + 1;
		}
		else
		{
			scan++;
		}
	}

	region.end = scan;
	return region;
}

std::string html_parser::extractRegionContent(std::string &html, html_region &region)
{
	if (region.start >= region.end || region.end > html.size())
		return "";

	int tagStart = region.start;
	int tagEnd = findTagEnd(html, tagStart);
	if (tagEnd == std::string::npos || tagEnd >= region.end)
		return "";

	return html.substr(tagEnd + 1, region.end - tagEnd - 1);
}

std::vector<parsed_link> html_parser::extractLinks(std::string &html)
{
	std::vector<parsed_link> links;
	int count = 0;

	char *skipTexts[] = { "skip", "search", "log in", "sign in", "subscribe", "menu", "home", "back", "next", "previous" };

	int pos = 0;
	while (pos < html.size())
	{
		int aPos = html.find("<a ", pos);
		if (aPos == std::string::npos)
			break;

		int tagEnd = findTagEnd(html, aPos);
		if (tagEnd == std::string::npos)
		{
			pos = aPos + 1;
			continue;
		}

		std::string href_name = "href";
		std::string href = html_parser::getAttribute(html, aPos, href_name);
		if (href.empty() || href[0] == '#' || href.find("javascript:") != std::string::npos)
		{
			pos = tagEnd + 1;
			continue;
		}

		std::string linkText;
		int closePos = html.find("</a>", tagEnd);
		if (closePos != std::string::npos)
		{
			linkText = html.substr(tagEnd + 1, closePos - tagEnd - 1);

			int innerPos = 0;
			while ((innerPos = linkText.find("<", innerPos)) != std::string::npos)
			{
				int innerEnd = linkText.find(">", innerPos);
				if (innerEnd == std::string::npos)
					break;
				linkText.erase(innerPos, innerEnd - innerPos + 1);
			}
		}

		while (!linkText.empty() && std::isspace((unsigned char)linkText[0]))
		{
			linkText.erase(linkText.begin());
		}
		while (!linkText.empty() && std::isspace((unsigned char)linkText.back()))
		{
			linkText.pop_back();
		}

		if (linkText.length() < 2)
		{
			pos = tagEnd + 1;
			continue;
		}

		std::string lowerText = toLower(linkText);
		bool skip = false;
		for (int i = 0; i < 10; i++)
		{
			if (lowerText.find(skipTexts[i]) != std::string::npos && linkText.length() < 30)
			{
				skip = true;
				break;
			}
		}
		if (skip)
		{
			pos = tagEnd + 1;
			continue;
		}

		count++;
		parsed_link link;
		link.number = count;
		link.text = linkText;
		link.url = href;
		link.position = aPos;
		links.push_back(link);

		pos = closePos != std::string::npos ? closePos + 4 : tagEnd + 1;
	}

	return links;
}

std::string removeBlocks(std::string &html, std::vector<std::string> &blockTags)
{
	std::string result = html;

	for (int t = 0; t < blockTags.size(); t++)
	{
		std::string openTag = "<" + blockTags[t];
		std::string closeTag = "</" + blockTags[t] + ">";

		int pos = 0;
		while ((pos = toLower(result).find(toLower(openTag), pos)) != std::string::npos)
		{
			int start = result.find("<", pos);
			int end = toLower(result).find(toLower(closeTag), pos);

			if (end == std::string::npos)
			{
				end = start + 100;
			}
			else
			{
				end += closeTag.length();
			}

			if (start != std::string::npos && end != std::string::npos)
			{
				result.erase(start, end - start);
				pos = start;
			}
			else
			{
				break;
			}
		}
	}

	return result;
}

std::string removeByClass(std::string &html, std::vector<std::string> &classNames)
{
	std::string result = html;

	for (int c = 0; c < classNames.size(); c++)
	{
		std::string search = "class=\"" + classNames[c];

		int pos = 0;
		while ((pos = result.find(search, pos)) != std::string::npos)
		{
			int start = result.rfind("<", pos);
			if (start == std::string::npos)
				break;

			// extract tag name from opening tag (handles <div>, <aside>, <nav>, etc.)
			int tagNameStart = start + 1;
			int tagNameEnd = result.find_first_of(" >\t\n", tagNameStart);
			if (tagNameEnd == std::string::npos)
				break;
			std::string tagName = result.substr(tagNameStart, tagNameEnd - tagNameStart);

			std::string openTag = "<" + tagName;
			std::string closeTag = "</" + tagName + ">";

			// count nested same-tag depth
			int scan = start;
			int depth = 1;
			int end = std::string::npos;
			while (scan < result.size() && depth > 0)
			{
				int nextOpen = toLower(result).find(toLower(openTag), scan + 1);
				int nextClose = toLower(result).find(toLower(closeTag), scan + 1);

				if (nextClose != std::string::npos && (nextOpen == std::string::npos || nextClose < nextOpen))
				{
					depth--;
					if (depth == 0)
					{
						end = nextClose + closeTag.length();
						break;
					}
					scan = nextClose + closeTag.length();
				}
				else if (nextOpen != std::string::npos)
				{
					depth++;
					scan = nextOpen + openTag.length();
				}
				else
				{
					break;
				}
			}

			if (end == std::string::npos || end <= start)
				end = pos + 100;

			if (end > start)
			{
				result.erase(start, end - start);
				pos = start;
			}
			else
			{
				break;
			}
		}
	}

	return result;
}

std::string findContentArea(std::string &html)
{
	char *selectors[] = { "itemprop=\"articleBody\"", "class=\"article-body\"", "class=\"entry-content\"", "class=\"post-content\"", "class=\"article-content\"", "id=\"content\"", "id=\"main\"", "role=\"main\"" };

	for (int i = 0; i < 8; i++)
	{
		int pos = html.find(selectors[i]);
		if (pos != std::string::npos)
		{
			int tagStart = html.rfind("<", pos);
			int tagEnd = html.find(">", pos);

			if (tagStart != std::string::npos && tagEnd != std::string::npos)
			{
				html_region region = html_parser::findRegionAt(html, tagStart);
				if (region.end > region.start)
				{
					return html_parser::extractRegionContent(html, region);
				}
			}
		}
	}

	int mainStart = html.find("<main");
	if (mainStart != std::string::npos)
	{
		html_region region = html_parser::findRegionAt(html, mainStart);
		if (region.end > region.start)
		{
			return html_parser::extractRegionContent(html, region);
		}
	}

	int articleStart = html.find("<article");
	if (articleStart != std::string::npos)
	{
		html_region region = html_parser::findRegionAt(html, articleStart);
		if (region.end > region.start)
		{
			return html_parser::extractRegionContent(html, region);
		}
	}

	int bodyStart = html.find("<body");
	int bodyEnd = html.find("</body>");
	if (bodyStart != std::string::npos && bodyEnd != std::string::npos)
	{
		int contentStart = html.find(">", bodyStart) + 1;
		return html.substr(contentStart, bodyEnd - contentStart);
	}

	return html;
}

std::string extractTitle(std::string &html)
{
	int titleStart = html.find("<title");
	if (titleStart != std::string::npos)
	{
		int openEnd = html.find(">", titleStart);
		int titleEnd = html.find("</title>", titleStart);
		/* both bounds must exist AND nest correctly: a missing ">" would
		 * wrap npos+1 to 0 and return the whole page head as the title */
		if (openEnd != std::string::npos && titleEnd != std::string::npos && openEnd < titleEnd)
		{
			return html.substr(openEnd + 1, titleEnd - openEnd - 1);
		}
	}

	int h1Pos = html.find("<h1");
	if (h1Pos != std::string::npos)
	{
		html_region region = html_parser::findRegionAt(html, h1Pos);
		if (region.end > region.start)
		{
			std::string content = html_parser::extractRegionContent(html, region);
			int innerPos = 0;
			while ((innerPos = content.find("<", innerPos)) != std::string::npos)
			{
				int innerEnd = content.find(">", innerPos);
				if (innerEnd == std::string::npos)
					break;
				content.erase(innerPos, innerEnd - innerPos + 1);
			}
			return content;
		}
	}

	return "";
}

std::string extractMetaContent(std::string &html, std::string &name)
{
	std::string search = "name=\"" + name + "\"";
	int pos = html.find(search);

	if (pos == std::string::npos)
	{
		search = "property=\"" + name + "\"";
		pos = html.find(search);
	}

	if (pos == std::string::npos)
		return "";

	int contentStart = html.find("content=\"", pos);
	char quote = '"';
	if (contentStart == std::string::npos)
	{
		contentStart = html.find("content='", pos);
		quote = '\'';
	}

	if (contentStart == std::string::npos)
		return "";

	contentStart += 9;
	int contentEnd = html.find(quote, contentStart);
	if (contentEnd == std::string::npos)
		return "";

	return html.substr(contentStart, contentEnd - contentStart);
}