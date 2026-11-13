# Native Linux x86-64 build for the unified v7 source tree. Run from the repo root.
# The Windows build is build.bat; both drivers assemble the same src/asm sources.
NASM ?= nasm
NASM_DEFINES ?= -DV7_LINUX
OUT ?= bin-linux
CPPFLAGS += -D_GNU_SOURCE -Isrc/linux/compat -Isrc/linux -Isrc -Itools -Itools/zydis_stub
CFLAGS ?= -O2
CXXFLAGS ?= -O2
HOST_LIBS = -ldl -lm -pthread
VALIDATORS = add_inplace attention bicubic config_probe encode execute fused generate matmul mdb memcpy quantize readwrite rmsnorm sampler sectorlogo sprintf stream swiglu vit_linear
CHECK_BINS = $(addprefix $(OUT)/v_,$(addsuffix _validate,$(VALIDATORS)))
MODELFREE_CHECK_BINS = $(OUT)/v_bicubic_validate $(OUT)/v_encode_validate $(OUT)/v_execute_validate $(OUT)/v_mdb_validate $(OUT)/v_readwrite_validate $(OUT)/v_sprintf_validate
LIB = $(OUT)/libv7_threading.so
HEADERS = src/v7_lib.h $(wildcard src/linux/compat/*.h)
RAG_TOOL = examples/rag/bm25_rag

.PHONY: all check clean
all: $(LIB) $(OUT)/build_db $(OUT)/inference_example $(OUT)/run_sectorlogo $(OUT)/echo_args $(OUT)/v7_server

# Without MODEL, check.sh runs only the model-free validators, so only those
# six are built; a MODEL builds + runs the full set (the sectorlogo gc-gate
# FFI suites additionally need slffi_test.so, and the RAG e2e suites spawn
# the bm25_rag helper at examples/rag/bm25_rag).
ifeq ($(MODEL),)
CHECK_PREREQS := $(MODELFREE_CHECK_BINS)
else
CHECK_PREREQS := $(CHECK_BINS) $(OUT)/slffi_test.so $(RAG_TOOL)
endif
check: all $(CHECK_PREREQS)
	python3 src/check_asm_comments.py
	bash src/linux/check.sh "$(abspath $(OUT))" "$(MODEL)"

$(OUT):
	mkdir -p "$@"

$(OUT)/v7.o: $(wildcard src/asm/*.asm src/asm/*.inc) | $(OUT)
	@if ! command -v $(NASM) >/dev/null 2>&1; then \
		echo 'NASM "$(NASM)" not found on PATH -- install nasm (see README, Building on Linux)'; exit 2; fi
	$(NASM) -f elf64 -O0 -w-other -w-prefix-seg -w-number-overflow $(NASM_DEFINES) -Isrc/asm/ src/asm/v7_threading.asm -o "$@"

$(OUT)/exports.map: src/v7_threading.def | $(OUT)
	LC_ALL=C awk 'BEGIN { print "{ global:" } { sub(/\r$$/,""); sub(/;.*/,""); if (NF && $$1 != "LIBRARY" && $$1 != "EXPORTS") print "    " $$1 ";" } END { print "  local: *; };" }' "$<" > "$@"

$(LIB): $(OUT)/v7.o src/linux/platform.c $(OUT)/exports.map
	$(CC) $(CFLAGS) -fPIC -shared -Wl,-z,defs,-z,text,-z,noexecstack,-Bsymbolic \
	    -Wl,--version-script="$(OUT)/exports.map",-soname,libv7_threading.so \
	    src/linux/platform.c "$(OUT)/v7.o" -o "$@" -pthread -ldl

$(OUT)/build_db: tools/db_build_tool.c src/linux/ini.h $(HEADERS) | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) "$<" -Wl,-rpath,'$$ORIGIN' -o "$@" $(HOST_LIBS)

$(OUT)/inference_example: examples/c_inference/inference_example/inference_example.c $(HEADERS) | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) "$<" -Wl,-rpath,'$$ORIGIN' -o "$@" $(HOST_LIBS)

$(OUT)/run_sectorlogo: examples/sectorlogo/run_sectorlogo.c $(HEADERS) | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) "$<" -rdynamic -Wl,-rpath,'$$ORIGIN' -o "$@" $(HOST_LIBS)

$(OUT)/echo_args: examples/sectorlogo/echo_args.c | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) "$<" -Wl,-rpath,'$$ORIGIN' -o "$@" $(HOST_LIBS)

# BM25 retrieval helper the SectorLogo RAG wrappers spawn via v7_execute
# ("examples/rag/bm25_rag.exe"; the POSIX spawn path strips the .exe suffix).
# Standalone C++: no engine headers, no engine link.
$(RAG_TOOL): examples/rag/bm25_rag.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 "$<" -o "$@"

$(OUT)/v_sampler_validate: CXXFLAGS += -maes
$(OUT)/v_sectorlogo_validate: CXXFLAGS += -msse4.2
$(OUT)/v_rmsnorm_validate: CXXFLAGS += -O0

# FFI test endpoints for the sectorlogo validator's CallLibrary suites; the
# functions carry the ms_abi attribute so the engine's Windows-convention
# marshalling reaches them with the right registers
$(OUT)/slffi_test.so: tools/slffi_test_dll.c | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared "$<" -o "$@"

$(OUT)/v_%_validate: tools/v_%_validate.cpp $(HEADERS) $(wildcard tools/*.h) | $(OUT)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 "$<" -Wl,-rpath,'$$ORIGIN' -o "$@" $(HOST_LIBS)

clean:
	rm -f -- "$(LIB)" "$(OUT)/v7.o" "$(OUT)/exports.map" "$(OUT)/build_db" \
	    "$(OUT)/inference_example" "$(OUT)/run_sectorlogo" "$(OUT)/echo_args" "$(OUT)/v7_server" "$(OUT)/dbg_image_recog" \
	    "$(OUT)/slffi_test.so" $(CHECK_BINS) "$(RAG_TOOL)"

SERVER_SOURCES = $(wildcard examples/server/*.cpp)

$(OUT)/v7_server: $(SERVER_SOURCES) $(wildcard examples/server/*.h) src/linux/server_linux.h src/linux/ini.h $(HEADERS) | $(OUT)
	$(CXX) $(CPPFLAGS) -I. $(CXXFLAGS) -std=c++17 $(SERVER_SOURCES) -o "$@" -Wl,-rpath,'$$ORIGIN' $(HOST_LIBS)

$(OUT)/dbg_image_recog: examples/c_inference/dbg_image_recog/dbg_image_recog.c $(HEADERS) | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) "$<" -Wl,-rpath,'$$ORIGIN' -o "$@" $(HOST_LIBS)
