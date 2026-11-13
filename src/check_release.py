#!/usr/bin/env python3
# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)
"""
check_release.py -- release hygiene gate (zero tolerance).

Checks, grown from the 2.28.0/2.29.0 audits:
  1. No tracked/shipped binaries outside the explicit fixture allow-list
     (no-git mode walks the tree and applies .gitignore, so build outputs
     never trip the gate).
  2. Every tools/*.c|cpp and examples/**/*.c|cpp is built by build.bat
     (or listed in the allow-list).
  3. Markdown links resolve; backticked file references in the live docs
     resolve by full path or basename; dev-only locations skipped.
  4. No mojibake, valid UTF-8 everywhere, .bat files CRLF-only.
  5. GPL header present in every .c/.cpp/.h/.inc/.asm.
  6. Known-stale quoted counts/phrasings absent from live docs.
  7. CHANGELOG: files said to be deleted must not exist.
  8. No absolute dev paths outside the exempt set.
  9. Version identity across README / CHANGELOG top entry / FILE_ID.DIZ /
     src/asm/version.inc.
  10. src/v7_threading.def: no duplicate exports (LNK4197), CRLF-uniform.
  11. ```logo blocks in live docs use registered SectorLogo spellings only.

Exit 0 = all clean; exit 1 = any finding. Run from the repo root
(default) or pass a tree root as argv[1]. With no .git (unpacked archive)
the tree is walked and .gitignore rules are applied.
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys

BINARIO_EXT = (".exe", ".dll", ".lib", ".obj", ".pdb", ".db")
BINARIES_ALLOW = {
    "examples/rag/web_ragg/web_ragg.db",   # shipped RAG demo database (fixture)
    "src/nasmfmt/nasmfmt.exe",             # prebuilt comment-checker formatter (deliberate)
    "tools/nasm.exe",                      # pinned assembler, shipped per owner decision
    "Zydis.lib",                           # dev-only Zydis import lib (release ships the stub)
}
RELEASE_VERSION = "2.36.8"  # must match README, CHANGELOG top entry, FILE_ID.DIZ
ABS_PATH_RE = re.compile(r"[A-Za-z]:\\(?:project|Users)")
# files whose CONTENT is legitimately path-bearing or historical:
#   sync_release.bat / tools/write_validator.py -- dev-only tools whose job
#     is to point at the dev tree
#   docs/how_to_make_a_release.md -- dev release process doc
#   scripts/check_parens.py -- dev-side paren checker
ABS_PATH_EXEMPT = {
    "docs/AGENTS_dev.md", "scripts/sync_release.bat", "tools/write_validator.py",
    "docs/how_to_make_a_release.md", "scripts/check_parens.py",
}
# live documentation scanned for dangling backticked references (the review's
# own scan scope): historical devlogs/bug logs at the repo root and the dev
# handbook are records of past states and are not rewritten retroactively
LIVE_DOC_PREFIXES = ("README.md", "docs/", "examples/")
# Documentation that ships verbatim but describes completed experiment sessions.
# The temporary scripts/artifacts it names are provenance, not live dependencies,
# so these files are exempt from backtick/script-name resolution (markdown links
# inside live docs are still checked for every file not listed here).
HISTORICAL_DOCS = {
    "docs/AGENTS_dev.md",
    "docs/measurment_worklog_and_results_report.md",
    "docs/speed_fix_results.md",
    "docs/benchmarks.md",
    "docs/v7_speed_advantages.md",
}
BACKTICK_RE = re.compile(r"`([\w./\\-]+\.(?:bat|c|cpp|h|inc|asm|py|logo|md|txt|ini|html|json|diz|exe|dll|db|raw|meta|bin))`")
# system/toolchain files and annotated dev-only locations that legitimately
# do not exist inside the tree
BACKTICK_KNOWN = {"vcvarsall.bat", "nasm.exe", "cl.exe", "link.exe", "config.json", "tuner_state.txt",
                  "zydis-master/include/Zydis/Zydis.h",
                  # generated core artifacts referenced everywhere by name
                  "v7_threading.dll", "lz4ht.db", "build_db.exe", "cat.raw",
                  # release-only source (examples\v7_server.c ships only in the release tree)
                  "v7_server.exe",
                  # retired names that historical docs legitimately mention
                  "hello_inference.exe", "q_token.bin", "tok_emb.bin",
                  # dev-tree rag example tools (sources/exes outside the tracked set)
                  "mem_bridge.exe", "ragg.exe", "ragg_extended.exe", "thought_ragg.exe",
                  "bin/slffi_test.dll"}
# dev-tree tools and attic records that shipped docs legitimately mention
# (annotated as dev-only); they exist in the dev tree, not in the release
DEV_ONLY_BASENAMES = {
    "tune_all_shapes.bat", "run_autotune.bat", "loadtest_tuned_vs_baseline.bat",
    "bug_log.md", "eval_history.md", "box_test_ref.py", "cat_rag_devlog.md",
    "helion_devlog.md", "ardmark_devlog.md", "rag_operations_guide.md",
    "why_stream_rewriting_matters.md", "test_membridge.bat",
    "_test_vlms_quick.bat", "_test_decide_session.bat",
    "llava_onevision_integration_plan.md",
    "bounding_box_test.md", "rewriting_test_plan.md",
    "constant_tuning_plan.md", "internal_tuner_implementation_plan.md",
    "bench_qwen.bat", "AGENTS.md",
    # external-engine benchmark drivers: they live in git-ignored dev engine
    # directories (openvino/, llamacpp/, onnxrt/) and never ship
    "smoke_ov.bat", "bench_ov.bat",
    "smoke_llamacpp.bat", "bench_llamacpp.bat",
    "smoke_ort.bat", "bench_ort.bat",
}
DEV_ONLY_SEGMENTS = ("build_logs/", "new_tools/", "docs/plans/", "attic",
                     "research_logbook", "internal_tools/", "vision/",
                     "v7_vlm_ref/", "sectorlogo_c/")
# dev-only trees: archived/planning content and the dev-only C++ port never
# ship, so their hygiene is out of scope for this gate
SKIP_PREFIXES = ("attic/", "sectorlogo_c/")
SOURCES_ALLOW = {
    # near-duplicate example sources kept for reading; build.bat compiles the
    # canonical (renamed) copies from tools\ -- these ship as documentation
    "examples/c_inference/test_ardmark_rag/test_ardmark_rag.c",
    "examples/c_inference/test_chat_only/test_chat_only.c",
    "examples/c_inference/test_engine_death/test_engine_death.c",
    "examples/c_inference/test_generate_only/test_generate_only.c",
    "examples/c_inference/test_image_multiturn/test_image_multiturn.c",
    "examples/c_inference/test_kv_cache/test_kv_cache.c",
    "examples/c_inference/test_multiturn/test_multiturn.c",
    "examples/c_inference/test_multiturn_manual/test_multiturn_manual.c",
    # web_ragg demo: built by examples/rag/web_ragg/build_web_ragg.bat
    "examples/rag/web_ragg/html_parser.cpp",
    "examples/rag/web_ragg/http_client.cpp",
    "examples/rag/web_ragg/search.cpp",
    "examples/rag/web_ragg/simpledb.c",
    "examples/rag/web_ragg/text_shortener.cpp",
    "examples/rag/web_ragg/text_summarizer.cpp",
    "examples/rag/web_ragg/text_utils.cpp",
    "examples/rag/web_ragg/web_ragg.cpp",
    "examples/rag/web_ragg/webtool.cpp",
    # unity-include helper pulled in by other tools via #include "win.c"
    "tools/win.c",
}
GPL_MARK = "GPL v3"
TEXT_EXT = (".md", ".c", ".cpp", ".h", ".hpp", ".py", ".inc", ".asm",
            ".logo", ".ini", ".bat", ".html", ".diz", ".txt")
MOJIBAKE_RUNS = re.compile("[\u00c2\u00c3\u00ce][\u0080-\u00bf\u00a0-\u00ff\u2014]")
STALE_PHRASINGS = [
    "8 LLMs + 3 VLMs",
    "supports 8 LLMs",
    "tested with 8 LLMs",
    "9 LLMs + 3 VLMs",
    "2 Q8 embedding-layer failures",
    "n=151936 (embedding layer)",
    "KNOWN-LIMIT: float-vs-double",
    "non-greedy stochastic mismatches",
    "640 tests",
    "22 validators",
    "23 validators",
    "25 validators",
    "over 20 validators",
    "18 validators",
]
# SectorLogo spellings are NOT denied by list any more (P0-2, reviewer
# candidate): the old hard-coded denylist could never know about names the
# docs invent next. check11 now builds the ALLOWLIST at check time from the
# registry itself -- see collect_registered_logo_names() below.
# dev-only targets that legitimately do not exist in a release tree;
# references to them in prose are fine, markdown LINKS to them are not
DEV_ONLY_PREFIXES = ("attic/", "attic\\", "research_logbook/")


def load_ignore_rules(root):
    """Minimal .gitignore matcher for no-git mode (P0-1, third audit): a
    fresh unpack that has been BUILT once contains tools/*.exe, bin/*.dll,
    build_tmp/ etc. -- all matched by .gitignore. The gate must not flag a
    user's own build outputs as tracked binaries. Supports the pattern
    subset this repo uses: blank/# comments, trailing / dir rules, /-rooted
    names, dir/**/dir, and *.ext globs."""
    rules = []
    path = os.path.join(root, ".gitignore")
    if not os.path.isfile(path):
        return rules
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rules.append(line)
    return rules


def is_ignored(rel, rules):
    for r in rules:
        neg = r.startswith("!")
        pat = r[1:] if neg else r
        if pat.startswith("/"):
            pat = pat.lstrip("/")
            hit = fnmatch.fnmatch(rel, pat) or rel.startswith(pat.rstrip("/") + "/")
        elif pat.endswith("/"):
            hit = rel.startswith(pat)
        elif "/" in pat:
            hit = fnmatch.fnmatch(rel, pat)
        else:
            hit = fnmatch.fnmatch(os.path.basename(rel), pat) or \
                any(fnmatch.fnmatch(part, pat) for part in rel.split("/"))
        if hit:
            return not neg
    return False


def tracked_files(root):
    git_dir = os.path.join(root, ".git")
    if not os.path.isdir(git_dir):
        # No VCS (fresh unpack of the release archive): the gate must still
        # run -- build.bat calls it in Phase 1, and an unpack has no .git.
        # Walk the tree instead; SKIP_PREFIXES, junk dirs and everything
        # .gitignore would exclude (build outputs!) still apply -- EXCEPT
        # paths in BINARIES_ALLOW: those are force-added in git (the allow
        # list exists precisely because .gitignore would drop them), so the
        # ignore rules must never win over the allow list (P1-1: a fresh
        # unzip lost web_ragg.db / nasmfmt.exe and every backtick reference
        # to them "did not resolve").
        rules = load_ignore_rules(root)
        keep = []
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs
                       if x not in (".git", "weights", "build_tmp", "bin",
                                    "build_logs", "node_modules")]
            for f in files:
                rel = os.path.relpath(os.path.join(d, f), root).replace("\\", "/")
                if rel in BINARIES_ALLOW:
                    keep.append(rel)
                    continue
                if rel.startswith(SKIP_PREFIXES) or is_ignored(rel, rules):
                    continue
                keep.append(rel)
        return keep
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    keep = []
    for f in out:
        f = f.replace("\\", "/")
        if not f.strip() or f.startswith(SKIP_PREFIXES):
            continue
        if not os.path.isfile(os.path.join(root, f)):  # tracked but removed from the working tree (a sync exclusion)
            continue
        keep.append(f)
    return keep


def check_binaries(files, findings):
    for f in files:
        if f.lower().endswith(BINARIO_EXT) and f not in BINARIES_ALLOW:
            findings.append("check1 binary tracked outside the allow-list: %s" % f)


def check_fetch_scripts(files, root, findings):
    # A zero-byte fetch_*.bat ships as a silent no-op: every setup path tells
    # a fresh user to run it, and nothing happens with no message (this
    # shipped once). Any tracked fetch script must carry content.
    for f in files:
        if f.startswith("fetch_") and f.lower().endswith(".bat"):
            path = os.path.join(root, f)
            try:
                size = os.path.getsize(path)
            except OSError:
                findings.append("check11 fetch script unreadable: %s" % f)
                continue
            if size == 0:
                findings.append("check11 fetch script is empty: %s" % f)


def check_sources_built(files, root, findings):
    with open(os.path.join(root, "scripts", "build.bat"), "r",
              encoding="utf-8", errors="replace") as fh:
        build = fh.read()
    for f in files:
        if not (f.startswith("tools/") or f.startswith("examples/")):
            continue
        if not f.lower().endswith((".c", ".cpp")):
            continue
        base = os.path.basename(f)
        if f in SOURCES_ALLOW:
            continue
        # build.bat's for-loops list bare names without extension
        if os.path.splitext(base)[0] not in build:
            findings.append("check2 source not built by build.bat: %s" % f)


def iter_markdown(files, root):
    for f in files:
        if f.lower().endswith(".md"):
            yield f


def check_markdown_refs(files, root, findings):
    # basename index for backticked references (resolve by name anywhere)
    base_index = {os.path.basename(f): f for f in files}
    source_index = {os.path.basename(f) for f in files
                    if f.lower().endswith((".c", ".cpp"))}
    doc_files = [f for f in files
                 if any(f == p or f.startswith(p) for p in LIVE_DOC_PREFIXES)
                 and not f.lower().endswith((".png", ".jpg", ".jpeg", ".gif",
                                            ".bmp", ".ico", ".webp", ".svg",
                                            ".zip", ".exe", ".dll", ".bin",
                                            ".db", ".raw", ".meta"))]
    # historical records ship in full but name artifacts from completed
    # experiment sessions; those names are provenance, not live dependencies
    scan_docs = [f for f in doc_files
                 if f.replace("\\", "/") not in HISTORICAL_DOCS]
    for f in scan_docs:
        full_path = os.path.join(root, f)
        if not os.path.isfile(full_path):         # an excluded/removed doc leaves a walk entry behind
            continue
        text = open(full_path, encoding="utf-8",
                    errors="replace").read()
        d = os.path.dirname(f)
        for target in re.findall(r"\]\(([^)#\s]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            t = target.replace("\\", "/")
            if any(t.startswith(p) for p in DEV_ONLY_PREFIXES):
                findings.append("check3 link into dev-only tree: %s in %s" % (target, f))
                continue
            p = os.path.normpath(os.path.join(root, d, t))
            if not os.path.exists(p) and os.path.basename(t) not in base_index:
                findings.append("check3 broken link '%s' in %s" % (target, f))
        # backticked file references must resolve by full path or basename;
        # scanned only in the live doc set (P1-4). Built-binary extensions get
        # one extra chance: a reference is stale only when NO matching source
        # (.c/.cpp) exists under the tree -- v7_threading.dll, build_db.exe
        # and friends are generated artifacts whose sources live in the tree,
        # while test_chat_only.exe names a binary nothing has built for years.
        for m in BACKTICK_RE.finditer(text):
            n = m.group(1).replace("\\", "/")
            if n in BACKTICK_KNOWN or os.path.basename(n) in BACKTICK_KNOWN:
                continue
            if "attic" in n or "research_logbook" in n:
                continue  # annotated dev-only locations
            if any(seg in n for seg in DEV_ONLY_SEGMENTS):
                continue
            if os.path.basename(n) in DEV_ONLY_BASENAMES:
                continue
            if os.path.basename(n) in base_index:
                continue
            base_low = os.path.basename(n).lower()
            stem, ext = os.path.splitext(base_low)
            if ext in (".exe", ".dll") and not os.path.basename(n) in base_index:
                src_a = stem + ".c"
                src_b = stem + ".cpp"
                if src_a in base_index or src_b in base_index or src_a in source_index or src_b in source_index:
                    continue  # buildable artifact -- its source lives in the tree
            if ext in (".db", ".bin", ".raw", ".meta") and ("weights/" in n.lower()
                                                            or "build_logs/" in n.lower()
                                                            or "tuning/" in n.lower()):
                continue  # generated model-tree artifacts, never tracked
            p = os.path.normpath(os.path.join(root, d, n))
            if not os.path.exists(p):
                findings.append("check3 backticked file not in tree: '%s' in %s"
                                % (n, f))


LOGO_HEADS = ("Output", "Make", "Define", "DefineCommand", "Repeat",
              "IfElse", "Print", "PrintRaw", "Run", "Local", "Prepend",
              "Stop", "Continue", "Exit", "SetStream", "Halt", "Reject",
              "EnableTrace", "EnableColors", "EnableStrictNames")
LOGO_BUILTIN_WORDS = {"nil"}        # value constants the registry does not list
WORD_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _looks_like_logo(label, block):
    # A fence is a SectorLogo snippet if it is labelled ```logo, or if any
    # line STARTS with a PascalCase SectorLogo command head (case-sensitive:
    # the house style mandates PascalCase call spellings). The exact case is
    # what keeps C pseudocode (api.md "output[c_out_pos] = 0") and shell,
    # tree and diagram fences (README, vision.md) out; a copy-pasted
    # `Output Frobnicate 1` snippet still trips it.
    if label == "logo":
        return True
    for line in block.splitlines():
        first = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)", line)
        if first and first.group(1) in LOGO_HEADS:
            return True
    return False


def collect_registered_logo_names(root):
    # The allowlist IS the registry: every SL_TYPE_WORD and SL_TYPE_STRING
    # string in sectorlogo_*.inc (case folded -- the tokenizer folds A-Z to
    # a-z; the FFI type tags are registered as STRING atoms) plus the
    # stdlib names the shipped autoexec.logo Defines. Compared
    # case-insensitively because names are case-insensitive in the
    # interpreter (P0-2, reviewer candidate).
    names = set()
    inc = os.path.join(root, "src", "asm")
    for f in sorted(os.listdir(inc)):
        if f.startswith("sectorlogo_") and f.endswith(".inc"):
            for m in re.finditer(r'SL_TYPE_(?:WORD|STRING),\s*"([^"]+)"',
                                 open(os.path.join(inc, f), encoding="utf-8",
                                      errors="replace").read()):
                names.add(m.group(1).lower())
    autoexec = os.path.join(root, "autoexec.logo")
    if os.path.isfile(autoexec):
        for m in re.finditer(r'\bDefine(?:Command)?\s+"([A-Za-z0-9_]+)"',
                             open(autoexec, encoding="utf-8",
                                  errors="replace").read(), re.I):
            names.add(m.group(1).lower())
    return names


def check_logo_blocks(files, root, findings):
    # Every fenced block in the live docs that looks like a SectorLogo
    # snippet (any fence label -- README and docs use unlabeled fences) may
    # only use registered spellings (P0-2, reviewer candidate). Names are
    # compared case-insensitively; a Define/DefineCommand inside the same
    # block registers a user name for that block; string literals, :var
    # references, numbers and symbols are not names.
    registered = collect_registered_logo_names(root)
    doc_files = [f for f in files
                 if any(f == p or f.startswith(p) for p in LIVE_DOC_PREFIXES)
                 and f.replace("\\", "/") != "docs/AGENTS_dev.md"]
    for f in doc_files:
        text = open(os.path.join(root, f), encoding="utf-8",
                    errors="replace").read()
        for bi, (label, block) in enumerate(
                re.findall(r"```([A-Za-z0-9]*)\r?\n(.*?)```", text, re.S)):
            if not _looks_like_logo(label, block):
                continue
            defined = set(m.group(1).lower() for m in re.finditer(
                r'\bDefine(?:Command)?\s+"([A-Za-z0-9_]+)"', block, re.I))
            for pl in re.finditer(
                    r'\bDefine(?:Command)?\s+"[A-Za-z0-9_]+"\s*\[([^\]]*)\]',
                    block, re.I):
                # the parameter list defines its names too ([a b] / [:a :b])
                for w in WORD_TOKEN.findall(pl.group(1)):
                    defined.add(w.lstrip(":").lower())
            for line in block.splitlines():
                code = re.sub(r'"(?:[^"\\]|\\.)*"', " ", line)  # string literals first: a ';' inside a string is not a comment
                code = code.split(";")[0]                      # then comments
                code = re.sub(r':[A-Za-z0-9_]+', " ", code)    # :var references
                for w in WORD_TOKEN.findall(code):
                    lw = w.lower()
                    if lw in registered or lw in defined \
                            or lw in LOGO_BUILTIN_WORDS:
                        continue
                    findings.append(
                        "check11 unregistered SectorLogo name '%s' in %s "
                        "(block %d)" % (w, f, bi + 1))


def check_def_file(files, root, findings):
    # duplicate exports (LNK4197) and mixed EOL in the module-definition file
    # (P1-1, third audit)
    p = os.path.join(root, "src/v7_threading.def")
    if not os.path.isfile(p):
        findings.append("check10 src/v7_threading.def missing")
        return
    raw = open(p, "rb").read()
    bare_lf = 0
    for i, b in enumerate(raw):
        if b == 10 and (i == 0 or raw[i - 1] != 13):
            bare_lf += 1
    if bare_lf:
        findings.append("check10 .def has %d bare-LF line ending(s)" % bare_lf)
    names = []
    for line in raw.decode("ascii", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith(";") and not s.startswith(".") and " " not in s \
                and s not in ("EXPORTS", "LIBRARY"):
            names.append(s)
    seen = {}
    for n in names:
        seen[n] = seen.get(n, 0) + 1
    for n, c in sorted(seen.items()):
        if c > 1:
            findings.append("check10 .def exports '%s' %d times (LNK4197)" % (n, c))


def check_dev_paths(files, root, findings):
    # absolute dev paths must not leak outside the exempt set (P0-6): the dev
    # handbook, the dev-only sync tool and the release-process docs are
    # path-bearing by nature
    for f in files:
        if f.replace("\\", "/") in ABS_PATH_EXEMPT:
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in TEXT_EXT and ext not in (".c", ".cpp", ".h"):
            continue
        text = open(os.path.join(root, f), encoding="utf-8",
                    errors="replace").read()
        for i, line in enumerate(text.splitlines(), 1):
            if ABS_PATH_RE.search(line):
                findings.append("check8 absolute dev path in %s:%d" % (f, i))


def check_versions(files, root, findings):
    # one version identity across README, CHANGELOG top entry, FILE_ID.DIZ
    # and the ASM version string (P0-2)
    def read(p):
        try:
            return open(os.path.join(root, p), encoding="utf-8",
                        errors="replace").read()
        except OSError:
            return None
    readme = read("README.md")
    changelog = read("CHANGELOG.md")
    fileid = read("FILE_ID.DIZ")
    if readme and RELEASE_VERSION not in readme:
        findings.append("check9 README does not carry version %s" % RELEASE_VERSION)
    if changelog:
        top = changelog.split("## ", 2)
        entry = top[1] if len(top) > 1 else ""
        # dev maintains the changelog under a top ## [Unreleased] section
        # between releases (Sec.166); that top entry is legitimate and a
        # versioned entry must sit directly beneath it. In the dev tree the
        # version below [Unreleased] is the PREVIOUS release (RELEASE_VERSION
        # is bumped first and the retitle happens in the release tree), so
        # only the shape is enforced here; the release run re-checks it.
        if entry.strip().startswith("[Unreleased]"):
            if len(top) < 3 or not re.match(r"\[\d+\.\d+\.\d+\]", top[2].strip()):
                findings.append("check9 CHANGELOG has [Unreleased] but no versioned entry beneath it")
        elif not entry.strip().startswith("[" + RELEASE_VERSION + "]"):
            findings.append("check9 CHANGELOG top entry is not [%s]" % RELEASE_VERSION)
    if fileid and RELEASE_VERSION not in fileid:
        findings.append("check9 FILE_ID.DIZ does not carry version %s" % RELEASE_VERSION)
    vsrc = read("src/asm/version.inc")
    if vsrc and ('"%s"' % RELEASE_VERSION) not in vsrc:
        findings.append("check9 src/asm/version.inc does not bake %s" % RELEASE_VERSION)


def check_encoding(files, root, findings):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in TEXT_EXT:
            continue
        raw = open(os.path.join(root, f), "rb").read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            findings.append("check4 invalid UTF-8: %s" % f)
            continue
        hits = MOJIBAKE_RUNS.findall(text)
        if hits:
            findings.append("check4 mojibake sequences in %s (%d runs)" % (f, len(hits)))
        if ext == ".bat":
            if b"\r\n" in raw and (raw.count(b"\n") != raw.count(b"\r\n")):
                findings.append("check4 mixed line endings in .bat: %s" % f)
            elif b"\r\n" not in raw and raw:
                findings.append("check4 .bat without CRLF: %s" % f)
        elif raw and ext in (".md", ".py", ".c", ".cpp", ".h", ".inc",
                             ".asm", ".logo", ".html", ".ini", ".html"):
            # document/code files must be UNIFORM: all-CRLF or all-LF (P2-9).
            # A lone \r not followed by \n is a stray CR inside a line. Plain
            # .txt is exempt: RAG fixtures (user_facts.txt) carry deliberate
            # trailing CR+space padding as content, and git stores them -text.
            crlf = raw.count(b"\r\n")
            lf_only = raw.count(b"\n") - crlf
            stray_cr = raw.count(b"\r") - crlf
            if lf_only > 0 and crlf > 0:
                findings.append("check13 mixed line endings (LF+CRLF) in: %s" % f)
            if stray_cr > 0:
                findings.append("check13 stray CR (not CRLF) in: %s" % f)


def check_agents_twin(files, root, findings):
    # One source of truth per document (P0-2): the release AGENTS.md is the
    # canonical handbook; docs\AGENTS_release.md was its stale hand-maintained
    # twin and is removed. Its reappearance is a finding -- twins drift.
    twin = os.path.join(root, "docs", "AGENTS_release.md")
    if os.path.isfile(twin):
        findings.append("check14 stale AGENTS twin exists (AGENTS.md is canonical): docs/AGENTS_release.md")


def check_orphan_bins(files, root, findings):
    # Packaging hygiene (P0-3): no build output (.pyc) or other generated junk
    # inside tracked trees; no empty directories (zip bloat / accidental
    # artifacts). Real binaries are governed by check1's allow-list.
    for f in files:
        if f.lower().endswith(".pyc"):
            findings.append("check15 .pyc inside the tree: %s" % f)
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".git", "weights", "build_tmp", "bin",
                                                "build_logs", "node_modules", "attic",
                                                "sectorlogo_c", "scheme_internal", "new_tools",
                                                "zydis-master", "zycore-c-master", "tuning",
                                                ".vscode")]
        for x in dirs:
            full = os.path.join(d, x)
            try:
                if not any(os.scandir(full)):
                    rel = os.path.relpath(full, root).replace("\\", "/")
                    if rel == "opt_and_bug":
                        continue  # standing work-queue directory (review agents drop reports here) -- empty between batches is normal
                    findings.append("check15 empty directory in tree: %s" % rel)
            except OSError:
                pass


def check_bat_refs(files, root, findings):
    # Setup/build scripts must reference scripts and docs that EXIST (P0-4).
    # Scan .bat/.md/.txt for <name>.bat tokens; dev-only history locations and
    # the exempt files are skipped. Allow-list: system/toolchain files, the
    # deleted-scripts inventory AGENTS/how-to must keep naming, dev-only helper
    # names (DEV_ONLY_BASENAMES), and CHANGELOG (history record).
    BAT_REF_EXEMPT_FILES = {"docs/how_to_make_a_release.md",
                            "AGENTS.md", "scripts/sync_release.bat", "CHANGELOG.md"}
    BAT_REF_EXEMPT_FILES |= HISTORICAL_DOCS
    BAT_REF_ALLOW = {
        "build_v7.bat", "build_tools.bat", "build_all.bat", "build_examples.bat",
        "build_validators.bat", "build_v7_server.bat", "build_bayes_classify.bat",
        "build_test_multiturn.bat", "build_test_multiturn_manual.bat",
        "build_autotune.bat", "build_llava_db.bat", "build_vision_db.bat",
        "build_llama3_ref.bat", "build_llama_ref.bat",  # deleted-scripts inventory
        "build_smolvlm_db.bat",  # named as never-written in fetch history
        "server.bat", "start_server.bat", "stop_server.bat",  # generated at runtime docs
        "vcvarsall.bat", "vcvars64.bat", "vsdevcmd.bat",  # system/toolchain
    }
    allow = set(BAT_REF_ALLOW)
    for n in DEV_ONLY_BASENAMES:
        allow.add(n.lower())
        allow.add((n + ".bat").lower())
    allow.update({
        # deleted dev-only helper scripts, named as history in dev records/docs
        "_samp_det3.bat", "_samp_det10.bat", "_rebuild_qwen3_db.bat",
        "_run_bm25_bench.bat", "thought_ragg_sweep.bat",
        "loadtest_bestknown_vs_baseline.bat", "build_cpp.bat",
        "_test_logo_scripts.bat", "_test_think_limiter.bat",
        "_test_batch_fatal.bat", "assemble_scheme.bat",
        "_test_server_quick.bat", "_probe_thinking_style.bat",
        "_run_vision_current.bat",
        # session-scratch drivers named as provenance in REM comments
        "rebuild_matmul_dbs.bat", "run_paper_rag_b.bat",
        # external-engine benchmark drivers (git-ignored dev engine dirs)
        "bench_mnn.bat", "smoke_mnn.bat",
        "bench_mnn5.bat", "bench_llamacpp5.bat",  # current-generation drivers, same out-of-tree worktrees
        "bench_ov.bat", "bench_llamacpp.bat", "bench_ort.bat",
    })
    bat_basenames = {os.path.basename(x).lower() for x in files if x.endswith(".bat")}
    present = set(files)
    for f in files:
        rel = f.replace("\\", "/")
        if rel in BAT_REF_EXEMPT_FILES:
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".bat", ".md", ".txt"):
            continue
        text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"\b([A-Za-z0-9_.-]+\.bat)\b", text):
            name = m.group(1)
            low = name.lower()
            if low in allow:
                continue
            if low in bat_basenames:
                continue
            # word-wrap / hyphenated-prose artifacts: the token ENDS with a real
            # .bat basename ("stop-with-stop_server.bat", "ench_compare_...bat"
            # after a wrapped 'b') -- accept those
            if any(low.endswith(bn) for bn in bat_basenames):
                continue
            findings.append("check16 referenced .bat not in tree: '%s' in %s" % (name, f))


def check_asm_count(files, root, findings):
    # The checker owns the numbers (P1-7): the AGENTS.md source counts for
    # src\asm and for the compiled tool sources must match the tree. Patterns:
    # "(NNN sources: ...)" for asm (the count carries a colon + breakdown),
    # "(NNN tool sources...)" for the manifest-derived tool count.
    agents = os.path.join(root, "AGENTS.md")
    if not os.path.isfile(agents):
        return
    try:
        asm_files = [x for x in os.listdir(os.path.join(root, "src", "asm"))
                     if x.lower().endswith((".inc", ".asm"))]
    except OSError:
        return
    actual = len(asm_files)
    text = open(agents, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"\((\d+)\s+sources\b([^)]*)\)", text):
        if "asm" not in m.group(2).lower():
            continue  # not the asm-count claim (e.g. a different breakdown)
        claimed = int(m.group(1))
        if claimed != actual:
            findings.append("check17 AGENTS.md claims %d asm sources but src/asm has %d"
                            % (claimed, actual))
    # tool sources derived from the manifest (field-1 records under tools/)
    manifest = os.path.join(root, "src", "build_c_targets.txt")
    if os.path.isfile(manifest):
        tool_sources = set()
        for line in open(manifest, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src = line.split("|")[0]
            if src.startswith("tools/"):
                tool_sources.add(src)
        for m in re.finditer(r"\((\d+)\s+tool sources\b[^)]*\)", text):
            claimed = int(m.group(1))
            if claimed != len(tool_sources):
                findings.append("check17 AGENTS.md claims %d tool sources but the manifest has %d"
                                % (claimed, len(tool_sources)))


GPL_HEADER_EXEMPT = {
    # vendored third-party code with its own license (stb_image 2.30, public
    # domain / MIT): prepending our GPL header would misstate its terms
    "src/linux/stb_image.h",
}


def check_gpl_headers(files, root, findings):
    for f in files:
        if f.replace("\\", "/") in GPL_HEADER_EXEMPT:
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".c", ".cpp", ".h", ".inc", ".asm"):
            continue
        head = open(os.path.join(root, f), encoding="utf-8",
                    errors="replace").read(400)
        if GPL_MARK not in head:
            findings.append("check5 missing GPL header: %s" % f)


def check_stale_phrasings(files, root, findings):
    for f in iter_markdown(files, root):
        for ln, line in enumerate(open(os.path.join(root, f), encoding="utf-8",
                                       errors="replace"), 1):
            if "resolved" in line.lower() or "claim stale" in line.lower():
                continue  # historical quotes inside resolution records are fine
            for phrase in STALE_PHRASINGS:
                if phrase in line:
                    findings.append("check6 stale phrasing '%s' in %s:%d"
                                    % (phrase, f, ln))


def check_changelog(files, root, findings):
    cl = os.path.join(root, "CHANGELOG.md")
    if not os.path.exists(cl):
        findings.append("check7 CHANGELOG.md missing")
        return
    text = open(cl, encoding="utf-8", errors="replace").read()
    latest = text.split("\n## ", 2)[1] if "\n## " in text else text
    fileset = set(files)
    for m in re.finditer(r"`([\w./\\-]+\.(?:c|cpp|h|inc|asm|bat|exe|md|db|diz|py))`",
                         latest):
        name = m.group(1).replace("\\", "/")
        same_line = latest[max(0, latest.rfind("\n", 0, m.start())):latest.find("\n", m.end())]
        if "delet" in same_line.lower() and name in fileset:
            findings.append("check7 changelog says deleted but tracked: %s" % name)
    # sanity: files the latest entry names with a path must exist (best effort)


def check_dll_size_claim(files, root, findings):
    # The FILE_ID.DIZ size claim must track the real DLL (P1-3, reviewer
    # candidate): "A single ~NNN KB DLL (v7_threading.dll)" is compared
    # against the locally built bin/v7_threading.dll within a 3% tolerance.
    # Skipped when the DLL is not built (fresh clone).
    diz = os.path.join(root, "FILE_ID.DIZ")
    dll = os.path.join(root, "bin", "v7_threading.dll")
    if not os.path.isfile(diz) or not os.path.isfile(dll):
        return
    m = re.search(r"~(\d+)\s*KB DLL \(v7_threading\.dll\)",
                  open(diz, encoding="utf-8", errors="replace").read())
    if not m:
        findings.append("check12 FILE_ID.DIZ has no '~NNN KB DLL (v7_threading.dll)' claim")
        return
    claimed = int(m.group(1))
    real = os.path.getsize(dll) // 1024
    if abs(claimed - real) > max(3, real * 3 // 100):
        findings.append(
            "check12 FILE_ID.DIZ claims ~%d KB DLL but bin/v7_threading.dll is %d KB"
            % (claimed, real))


def check_manifest_completeness(files, root, findings):
    # check18 (GLM-20 rule 1): every output-producing compile line in
    # build.bat (cl + the step-3 dbg_inference nasm) must have a
    # src/build_c_targets.txt record, and every record must be produced by
    # some step. A missing record means the freshness gate can certify a
    # STALE binary as FRESH; a dead record means the gate can never report
    # FRESH and every build pays the full C/C++ rebuild.
    bat = os.path.join(root, "scripts", "build.bat")
    manifest = os.path.join(root, "src", "build_c_targets.txt")
    if not os.path.isfile(bat) or not os.path.isfile(manifest):
        findings.append("check18 scripts/build.bat or src/build_c_targets.txt missing")
        return
    bat_text = open(bat, encoding="utf-8", errors="replace").read()

    def norm(path):
        n = path.replace("%THIS_DIR%", "").replace("%TOOLS_DIR%", "tools")
        n = n.replace("%TOOL_SRC%", "tools").replace("%ASM_DIR%", "src/asm")
        n = n.replace("%BIN_DIR%", "bin").replace("\\", "/")
        return n.replace("//", "/").strip("/").lower()

    built_sources = set()
    for m in re.finditer(r"cl\s+.*?\"([^\"]+?\.(?:c|cpp))\"", bat_text):
        if "%%" in m.group(1):
            continue  # a for-loop body line -- handled by the loop scan below
        built_sources.add(norm(m.group(1)))
    for m in re.finditer(r"nasm\s+.*?\"([^\"]+?\.asm)\"", bat_text):
        src = norm(m.group(1))
        if not src.startswith("src/"):  # the DLL asm is rebuilt unconditionally, no manifest record by design
            built_sources.add(src)
    # loop bodies: every `for %%f in (a b c) do` whose file has a cl line
    # referencing %TOOLS_DIR%\%%f.<ext> contributes tools/<name>.<ext> records
    ext_m = re.search(r"cl\s+.*?%TOOLS_DIR%\\%%f\.(cpp|c)", bat_text)
    if ext_m:
        for m in re.finditer(r"for %%f in \(([^)]+)\)", bat_text):
            for name in [x.strip() for x in m.group(1).split()]:
                built_sources.add(("tools/" + name + "." + ext_m.group(1)).lower())

    manifest_sources = set()
    for line in open(manifest, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        manifest_sources.add(norm(line.split("|")[0]))

    for s in sorted(built_sources - manifest_sources):
        findings.append("check18 built by build.bat but has no manifest record: %s" % s)
    for s in sorted(manifest_sources - built_sources):
        findings.append("check18 manifest record built by no build.bat step: %s" % s)


def check_derived_counts(files, root, findings):
    # check19 (GLM-20 rule 2): numbers that docs hand-copied must be derived.
    # The export count comes from src/v7_threading.def, the resolve count from
    # V7_RESOLVE( lines in src/v7_lib.h; docs/api.md claims are compared
    # against them. The text-LLM count comes from the fetch_*.bat scripts.
    def_file = os.path.join(root, "src", "v7_threading.def")
    lib_file = os.path.join(root, "src", "v7_lib.h")
    api_file = os.path.join(root, "docs", "api.md")
    if os.path.isfile(def_file) and os.path.isfile(lib_file) and os.path.isfile(api_file):
        exports = set()
        for line in open(def_file, encoding="utf-8", errors="replace"):
            m = re.match(r"^\s*([A-Za-z_]\w*)\s*$", line)
            if m and not line.lstrip().startswith(("LIBRARY", "EXPORTS")):
                exports.add(m.group(1))
        resolves = len(re.findall(r"V7_RESOLVE\(", open(lib_file, encoding="utf-8", errors="replace").read()))
        api = open(api_file, encoding="utf-8", errors="replace").read()
        m = re.search(r"All\s+(\d+)\s+exports", api)
        if m and int(m.group(1)) != len(exports):
            findings.append("check19 docs/api.md claims %s exports, src/v7_threading.def has %d"
                            % (m.group(1), len(exports)))
        m = re.search(r"(\d+)\s+of them are resolved into", api)
        if m and int(m.group(1)) != resolves:
            findings.append("check19 docs/api.md claims %s resolved, src/v7_lib.h has %d V7_RESOLVE lines"
                            % (m.group(1), resolves))
    # text-LLM count vs docs claims: fetch_*.bat minus the vision/decide tools
    llm_fetches = 0
    for f in os.listdir(root):
        low = f.lower()
        if low.startswith("fetch_") and low.endswith(".bat"):
            stem = low[6:-4]
            if stem.startswith(("decider", "llava", "smolvlm", "qwen3_vl")):
                continue
            llm_fetches += 1
    if llm_fetches:
        for f in files:
            if not f.endswith(".md"):
                continue
            text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
            if "resolved" in text.lower() or "claim stale" in text.lower():
                continue
            for m in re.finditer(r"\b(\d+)\s+tested\s+LLMs", text):
                if int(m.group(1)) != llm_fetches:
                    findings.append("check19 %s claims %s tested LLMs, %d fetch_*.bat scripts define them"
                                    % (f, m.group(1), llm_fetches))
    # validator count vs present-tense doc claims: the count is derived from
    # the tools/v_*_validate.cpp sources. The changelog is exempt by name;
    # dated progress records are exempt per line (resolved / at the time /
    # passed / green markers -- the shapes historical pass-reports use).
    validator_count = 0
    tools_dir = os.path.join(root, "tools")
    if os.path.isdir(tools_dir):
        for f in os.listdir(tools_dir):
            if f.startswith("v_") and f.endswith("_validate.cpp"):
                validator_count += 1
    if validator_count:
        for f in files:
            low = f.lower()
            if low == "changelog.md" or not (low.endswith(".md") or low.endswith(".diz")):
                continue
            for i, line in enumerate(open(os.path.join(root, f), encoding="utf-8", errors="replace"), 1):
                low_line = line.lower()
                if "validators" not in low_line:
                    continue
                if any(w in low_line for w in ("resolved", "at the time", "passed", "green")):
                    continue
                for m in re.finditer(r"\b(\d+)\s+validators", line):
                    if int(m.group(1)) != validator_count:
                        findings.append("check19 %s:%d claims %s validators, tools/ has %d v_*_validate.cpp"
                                        % (f, i, m.group(1), validator_count))


def check_markdown_backtick_corruption(files, root, findings):
    # check20 (GLM-20 rule 4): docs/setup.md once lost every backtick to a
    # literal 'v' (vcl.exev, vNASM_DIRv, vvv fences) and the reference gates
    # went blind for the whole file. Flag the corruption shape: many
    # v...v pseudo-spans with almost no real backticks left.
    for f in files:
        if not f.endswith(".md"):
            continue
        text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        pseudo = len(re.findall(r"\bv[\w./\\-]{2,60}v\b", text))
        if pseudo >= 5 and text.count("`") <= 4:
            findings.append("check20 possible backtick corruption in %s (%d v..v pseudo-spans, %d real backticks)"
                            % (f, pseudo, text.count("`")))


def check_bat_file_deps(files, root, findings):
    # check16b (GLM-20 rule 5): root .bat scripts reference non-.bat files
    # (baselines, fixtures, INI, logo) that must exist in the tree -- the
    # class of bug where run_vision_smoke.bat hard-required
    # vision\vision_baseline.txt and check16's .bat/.exe scan could not see it.
    # Scope: shipped root bats. fetch_*.bat list REMOTE HF repo patterns,
    # test_*/_* bats are dev scratch -- both out of scope.
    tree_basenames = {os.path.basename(x).lower() for x in files}
    for f in files:
        if not f.endswith(".bat") or "/" in f.replace("\\", "/"):
            continue  # root bats only: nested bats resolve against their own directory
        base = os.path.basename(f).lower()
        if base.startswith(("fetch_", "test_", "_")):
            continue
        text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"([A-Za-z0-9_][A-Za-z0-9_\\/.-]*\.(?:txt|raw|meta|logo|diz))\b", text):
            tok = m.group(1).replace("\\", "/")
            if m.start() > 0 and text[m.start() - 1] in ("%\\"):
                continue  # glued to a cmd variable: the real path is built at runtime
            if any(tok.lower().startswith(d) for d in
                   ("openvino/", "llamacpp/", "onnxrt/", "mnn/")):
                continue  # dev-only external-engine directories, they never ship
            if any(x in tok for x in ("build_logs", "weights/", "models/", "_vtest", "_vllm", "bench_time",
                                      "hill_climb_state", "collect_1h", "shape_bench", "out.txt",
                                      "prompt_poem", "bm25_case", "bm25_bench.logo")):
                continue  # runtime artifacts / generated databases / variable-expanded paths
            if os.path.basename(tok).lower() in tree_basenames:
                continue
            p = os.path.normpath(os.path.join(root, tok))
            if not os.path.exists(p):
                findings.append("check16b %s references missing file: %s" % (f, m.group(1)))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    files = tracked_files(root)
    findings = []
    check_binaries(files, findings)
    check_fetch_scripts(files, root, findings)
    check_sources_built(files, root, findings)
    check_markdown_refs(files, root, findings)
    check_encoding(files, root, findings)
    check_gpl_headers(files, root, findings)
    check_stale_phrasings(files, root, findings)
    check_changelog(files, root, findings)
    check_dev_paths(files, root, findings)
    check_versions(files, root, findings)
    check_logo_blocks(files, root, findings)
    check_def_file(files, root, findings)
    check_dll_size_claim(files, root, findings)
    check_agents_twin(files, root, findings)
    check_orphan_bins(files, root, findings)
    check_bat_refs(files, root, findings)
    check_asm_count(files, root, findings)
    check_manifest_completeness(files, root, findings)
    check_derived_counts(files, root, findings)
    check_markdown_backtick_corruption(files, root, findings)
    check_bat_file_deps(files, root, findings)
    if findings:
        print("check_release: %d FINDING(S) -- all must be fixed:" % len(findings))
        for x in findings:
            print("  " + x)
        return 1
    print("check_release: ALL CHECKS PASSED (%d tracked files)" % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
