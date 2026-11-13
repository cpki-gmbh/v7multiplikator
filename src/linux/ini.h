// v7 - INI reader used by the Linux database builder
// License: GPL v3, see LICENSE
#pragma once
#include <ctype.h>

static char *trim_ini(char *value)
{
    while (isspace((unsigned char)*value))
        ++value;
    char *end = value + strlen(value);
    while (end > value && isspace((unsigned char)end[-1]))
        --end;
    *end = 0;
    return value;
}

static unsigned GetPrivateProfileStringA(const char *section, const char *key,
                                          const char *fallback, char *buffer,
                                          unsigned capacity, const char *path)
{
    if (!capacity)
        return 0;
    snprintf(buffer, capacity, "%s", fallback);
    FILE *file = fopen(path, "r");
    if (!file)
        return strlen(buffer);
    char *line = NULL;
    size_t line_capacity = 0;
    int in_section = 0;
    while (getline(&line, &line_capacity, file) >= 0)
    {
        char *entry = trim_ini(line);
        if (*entry == '[')
        {
            char *end = strchr(entry, ']');
            if (end)
                *end = 0;
            in_section = end && !strcasecmp(trim_ini(entry + 1), section);
        }
        else if (in_section && *entry != ';' && *entry != '#')
        {
            char *equals = strchr(entry, '=');
            if (!equals)
                continue;
            *equals = 0;
            if (strcasecmp(trim_ini(entry), key))
                continue;
            char *value = trim_ini(equals + 1);
            size_t length = strlen(value);
            if (length >= 2 && (*value == '"' || *value == '\'') && value[length - 1] == *value)
            {
                value[length - 1] = 0;
                ++value;
            }
            snprintf(buffer, capacity, "%s", value);
            break;
        }
    }
    free(line);
    fclose(file);
    return strlen(buffer);
}

static int GetPrivateProfileIntA(const char *section, const char *key, int fallback, const char *path)
{
    char value[64];
    GetPrivateProfileStringA(section, key, "", value, sizeof(value), path);
    char *end;
    long result = strtol(value, &end, 10);
    return end == value ? fallback : (int)result;
}

static unsigned GetFullPathNameA(const char *path, unsigned capacity, char *buffer, void *unused)
{
    (void)unused;
    char *absolute = realpath(path, NULL);
    if (!absolute)
        return 0;
    size_t length = strlen(absolute);
    if (length < capacity)
        memcpy(buffer, absolute, length + 1);
    free(absolute);
    return length < capacity ? (unsigned)length : 0;
}
