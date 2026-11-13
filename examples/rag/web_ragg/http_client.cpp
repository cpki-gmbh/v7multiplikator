// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#include "http_client.h"
#include <windows.h>
#include <cstdio>

http_response curlGet(std::string &url, int timeoutSec)
{
	http_response resp;
	resp.statusCode = 0;

	std::string cmd = "curl -sL -w \"\\n%{http_code}\\n%{url_effective}\" ";
	cmd += "-A \"Lynx/2.8.9dev.16 libwww-FM/2.14 SSL-MM/1.4.1\" ";
	cmd += "--max-time " + std::to_string(timeoutSec) + " ";
	cmd += "-H \"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\" ";
	cmd += "-H \"Accept-language: en-US,en;q=0.5\" ";
	cmd += "-H \"Accept-charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7\" ";
	cmd += "-H \"Referer: https://www.bing.com/\" ";
	cmd += "-H \"Connection: keep-alive\" ";
	cmd += "--compressed ";
	// escape embedded double quotes (cmd.exe uses "" not \")
	std::string escapedUrl = url;
	int qpos = 0;
	while ((qpos = escapedUrl.find("\"", qpos)) != std::string::npos)
	{
		escapedUrl.replace(qpos, 1, "\"\"");
		qpos += 2;
	}
	cmd += "\"" + escapedUrl + "\"";

	int cmdLen = MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, 0, 0);
	wchar_t *wcmd = new wchar_t[cmdLen];
	MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, wcmd, cmdLen);

	FILE *pipe = _wpopen(wcmd, L"r");
	delete[] wcmd;

	if (!pipe)
	{
		resp.error = "Failed to run curl";
		return resp;
	}

	char buffer[4096];
	std::string output;
	while (fgets(buffer, sizeof(buffer), pipe) != 0)
	{
		output += buffer;
	}

	_pclose(pipe);

	int lastNewline = output.rfind('\n');
	int secondLast = (lastNewline > 0) ? output.rfind('\n', lastNewline - 1) : std::string::npos;

	if (lastNewline != std::string::npos && secondLast != std::string::npos)
	{
		std::string statusStr = output.substr(secondLast + 1, lastNewline - secondLast - 1);
		std::string finalUrl = output.substr(lastNewline + 1);
		resp.statusCode = atoi(statusStr.c_str());
		resp.finalUrl = finalUrl;
		resp.body = output.substr(0, secondLast);
	}
	else if (lastNewline != std::string::npos)
	{
		resp.statusCode = atoi(output.substr(lastNewline + 1).c_str());
		resp.body = output.substr(0, lastNewline);
	}
	else
	{
		resp.body = output;
	}

	return resp;
}

std::string curlGetBody(std::string &url, int timeoutSec)
{
	http_response resp = curlGet(url, timeoutSec);
	return resp.body;
}

int curlGetStatusCode(std::string &url, int timeoutSec)
{
	std::string cmd = "curl -sL -o NUL -w \"%{http_code}\" ";
	cmd += "-A \"Lynx/2.8.9dev.16 libwww-FM/2.14 SSL-MM/1.4.1\" ";
	cmd += "--max-time " + std::to_string(timeoutSec) + " ";
	// escape embedded double quotes to prevent command injection
	std::string escapedUrl = url;
	int qpos = 0;
	while ((qpos = escapedUrl.find("\"", qpos)) != std::string::npos)
	{
		escapedUrl.replace(qpos, 1, "\"\"");
		qpos += 2;
	}
	cmd += "\"" + escapedUrl + "\"";

	int cmdLen = MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, 0, 0);
	wchar_t *wcmd = new wchar_t[cmdLen];
	MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, wcmd, cmdLen);

	FILE *pipe = _wpopen(wcmd, L"r");
	delete[] wcmd;

	if (!pipe)
	{
		return 0;
	}

	char buffer[32];
	std::string output;
	if (fgets(buffer, sizeof(buffer), pipe) != 0)
	{
		output = buffer;
	}

	_pclose(pipe);

	return atoi(output.c_str());
}