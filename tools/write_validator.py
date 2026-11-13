#!/usr/bin/env python3
"""Write v_sectorlogo_validate.cpp"""
import os

out = r"C:\project\v7\tools\v_sectorlogo_validate.cpp"

content = r"""/* v_sectorlogo_validate.cpp -- SectorLogo64 interpreter comprehensive validator
 *
 * Tests v7s_sectorlogo_interp() end-to-end, covering all language features.
 * Takes <dll_path> as first positional arg.
 * Uses Zydis for crash disassembly (via zydis_debug_header.h).
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define SL_CONS_TAG      0x80000000
#define SL_NUMBER_TAG    0x40000000
#define SL_TAG_MASK      0xC0000000
#define SL_PAYLOAD_MASK  0x3FFFFFFF

 int num_val(int t) {
    int p = t & SL_PAYLOAD_MASK;
    if (p & 0x20000000) return p | 0xC0000000;
    return p;
}
 int is_num(int t) { return (t & SL_TAG_MASK) == SL_NUMBER_TAG; }
 int is_nil(int t) { return t == 0; }

 int g_pass = 0, g_fail = 0;
 v7_lib* g_lib = nullptr;

 int run( char* script) {
    return g_lib->v7s_sectorlogo_interp(script);
}

 void test_num( char* desc,  char* script, int expected) {
    int r = run(script);
    if (is_num(r) && num_val(r) == expected) {
        g_pass++; printf("  PASS: %s -> %d\n", desc, expected);
    } else {
        g_fail++;
        if (is_num(r)) printf("  FAIL: %s -> got %d (0x%08X), expected %d\n", desc, num_val(r), (unsigned)r, expected);
        else printf("  FAIL: %s -> got non-NUMBER 0x%08X, expected %d\n", desc, (unsigned)r, expected);
    }
}

 void test_true( char* desc,  char* script) {
    int r = run(script);
    if (r != 0) { g_pass++; printf("  PASS: %s -> TRUE\n", desc); }
    else { g_fail++; printf("  FAIL: %s -> got NIL, expected TRUE\n", desc); }
}

 void test_nil( char* desc,  char* script) {
    int r = run(script);
    if (is_nil(r)) { g_pass++; printf("  PASS: %s -> NIL\n", desc); }
    else { g_fail++; printf("  FAIL: %s -> got 0x%08X, expected NIL\n", desc, (unsigned)r); }
}

 void test_atom( char* desc,  char* script) {
    int r = run(script);
    int tag = r & SL_TAG_MASK;
    if (r != 0 && tag == 0) { g_pass++; printf("  PASS: %s -> atom\n", desc); }
    else { g_fail++; printf("  FAIL: %s -> got 0x%08X, expected atom\n", desc, (unsigned)r); }
}

 void test_cons( char* desc,  char* script) {
    int r = run(script);
    if (r & SL_CONS_TAG) { g_pass++; printf("  PASS: %s -> cons\n", desc); }
    else { g_fail++; printf("  FAIL: %s -> got 0x%08X, expected cons\n", desc, (unsigned)r); }
}

/* ====== TEST SUITES ====== */

 void test_integer_literals() {
    printf("\n--- Integer literals ---\n");
    test_num("0", "OUTPUT 0", 0);
    test_num("1", "OUTPUT 1", 1);
    test_num("42", "OUTPUT 42", 42);
    test_num("12345", "OUTPUT 12345", 12345);
    test_num("-1", "OUTPUT -1", -1);
    test_num("-42", "OUTPUT -42", -42);
    test_num("999999", "OUTPUT 999999", 999999);
}

 void test_prefix_arithmetic() {
    printf("\n--- Prefix arithmetic ---\n");
    test_num("SUM 2 3", "OUTPUT SUM 2 3", 5);
    test_num("SUM -5 10", "OUTPUT SUM -5 10", 5);
    test_num("DIFF 10 3", "OUTPUT DIFF 10 3", 7);
    test_num("DIFF 3 10", "OUTPUT DIFF 3 10", -7);
    test_num("DIFFERENCE 20 5", "OUTPUT DIFFERENCE 20 5", 15);
    test_num("PRODUCT 6 7", "OUTPUT PRODUCT 6 7", 42);
    test_num("PRODUCT -3 8", "OUTPUT PRODUCT -3 8", -24);
    test_num("QUOTIENT 42 6", "OUTPUT QUOTIENT 42 6", 7);
    test_num("QUOTIENT 7 2", "OUTPUT QUOTIENT 7 2", 3);
    test_num("DIVIDE 20 4", "OUTPUT DIVIDE 20 4", 5);
    test_num("REMAINDER 7 2", "OUTPUT REMAINDER 7 2", 1);
    test_num("MODULO 10 3", "OUTPUT MODULO 10 3", 1);
    test_num("MOD 10 3", "OUTPUT MOD 10 3", 1);
    test_num("POWER 2 10", "OUTPUT POWER 2 10", 1024);
    test_num("POWER 3 3", "OUTPUT POWER 3 3", 27);
    test_num("POWER 5 0", "OUTPUT POWER 5 0", 1);
    test_num("SQRT 49", "OUTPUT SQRT 49", 7);
    test_num("SQRT 100", "OUTPUT SQRT 100", 10);
    test_num("SQRT 0", "OUTPUT SQRT 0", 0);
    test_num("ABS 42", "OUTPUT ABS 42", 42);
    test_num("ABS -42", "OUTPUT ABS -42", 42);
    test_num("MINUS 5", "OUTPUT MINUS 5", -5);
    test_num("MINUS -3", "OUTPUT MINUS -3", 3);
    test_num("NEG 8", "OUTPUT NEG 8", -8);
    test_num("MAX 3 7", "OUTPUT MAX 3 7", 7);
    test_num("MAX 7 3", "OUTPUT MAX 7 3", 7);
    test_num("MIN 3 7", "OUTPUT MIN 3 7", 3);
    test_num("MIN 7 3", "OUTPUT MIN 7 3", 3);
}

 void test_infix_arithmetic() {
    printf("\n--- Infix arithmetic ---\n");
    test_num("2 + 3", "OUTPUT 2 + 3", 5);
    test_num("10 - 3", "OUTPUT 10 - 3", 7);
    test_num("6 * 7", "OUTPUT 6 * 7", 42);
    test_num("42 / 6", "OUTPUT 42 / 6", 7);
    test_num("7 / 2", "OUTPUT 7 / 2", 3);
    test_num("10 pct 3", "OUTPUT 10 % 3", 1);
    test_num("1 + 2 + 3", "OUTPUT 1 + 2 + 3", 6);
    test_num("2 + 3 * 4", "OUTPUT 2 + 3 * 4", 20);
}

 void test_divmod_edge() {
    printf("\n--- Division/modulo edge cases ---\n");
    test_num("QUOTIENT 5 0", "OUTPUT QUOTIENT 5 0", 0);
    test_num("REMAINDER 5 0", "OUTPUT REMAINDER 5 0", 0);
    test_num("5 / 0", "OUTPUT 5 / 0", 0);
    test_num("5 pct 0", "OUTPUT 5 % 0", 0);
}

 void test_prefix_bitwise() {
    printf("\n--- Prefix bitwise ---\n");
    test_num("BITAND 12 10", "OUTPUT BITAND 12 10", 8);
    test_num("BITOR 12 10", "OUTPUT BITOR 12 10", 14);
    test_num("BITXOR 12 10", "OUTPUT BITXOR 12 10", 6);
    test_num("BITNOT 0", "OUTPUT BITNOT 0", ~0);
    test_num("LSHIFT 1 4", "OUTPUT LSHIFT 1 4", 16);
    test_num("RSHIFT 16 2", "OUTPUT RSHIFT 16 2", 4);
    test_num("LSHIFT 7 3", "OUTPUT LSHIFT 7 3", 56);
}

 void test_infix_bitwise() {
    printf("\n--- Infix bitwise ---\n");
    test_num("12 band 10", "OUTPUT 12 & 10", 8);
    test_num("12 bor 10", "OUTPUT 12 | 10", 14);
    test_num("12 bxor 10", "OUTPUT 12 ^ 10", 6);
    test_num("1 shl 4", "OUTPUT 1 << 4", 16);
    test_num("16 shr 2", "OUTPUT 16 >> 2", 4);
}

 void test_prefix_relational() {
    printf("\n--- Prefix relational ---\n");
    test_true("EQUALP 3 3", "OUTPUT EQUALP 3 3");
    test_nil("EQUALP 3 4", "OUTPUT EQUALP 3 4");
    test_true("NOTEQUALP 3 4", "OUTPUT NOTEQUALP 3 4");
    test_nil("NOTEQUALP 3 3", "OUTPUT NOTEQUALP 3 3");
    test_true("GREATERP 5 3", "OUTPUT GREATERP 5 3");
    test_nil("GREATERP 3 5", "OUTPUT GREATERP 3 5");
    test_true("LESSP 3 5", "OUTPUT LESSP 3 5");
    test_nil("LESSP 5 3", "OUTPUT LESSP 5 3");
    test_true("GREATEREQUALP 5 5", "OUTPUT GREATEREQUALP 5 5");
    test_nil("GREATEREQUALP 3 5", "OUTPUT GREATEREQUALP 3 5");
    test_true("LESSEQUALP 5 5", "OUTPUT LESSEQUALP 5 5");
    test_nil("LESSEQUALP 5 3", "OUTPUT LESSEQUALP 5 3");
}

 void test_infix_relational() {
    printf("\n--- Infix relational ---\n");
    test_true("3 eq 3", "OUTPUT 3 = 3");
    test_nil("3 eq 4", "OUTPUT 3 = 4");
    test_true("3 ne 4", "OUTPUT 3 != 4");
    test_nil("3 ne 3", "OUTPUT 3 != 3");
    test_true("5 gt 3", "OUTPUT 5 > 3");
    test_nil("3 gt 5", "OUTPUT 3 > 5");
    test_true("3 lt 5", "OUTPUT 3 < 5");
    test_nil("5 lt 3", "OUTPUT 5 < 3");
    test_true("5 ge 5", "OUTPUT 5 >= 5");
    test_nil("4 ge 5", "OUTPUT 4 >= 5");
    test_true("5 le 5", "OUTPUT 5 <= 5");
    test_nil("5 le 4", "OUTPUT 5 <= 4");
}

 void test_boolean() {
    printf("\n--- Boolean operators ---\n");
    test_true("AND 1 1", "OUTPUT AND 1 1");
    test_nil("AND 1 0", "OUTPUT AND 1 0");
    test_nil("AND 0 1", "OUTPUT AND 0 1");
    test_nil("AND 0 0", "OUTPUT AND 0 0");
    test_true("OR 1 0", "OUTPUT OR 1 0");
    test_true("OR 0 1", "OUTPUT OR 0 1");
    test_nil("OR 0 0", "OUTPUT OR 0 0");
    test_true("NOT 0", "OUTPUT NOT 0");
    test_nil("NOT 1", "OUTPUT NOT 1");
    test_true("ZEROP 0", "OUTPUT ZEROP 0");
    test_nil("ZEROP 1", "OUTPUT ZEROP 1");
    test_true("NUMBERP 5", "OUTPUT NUMBERP 5");
    test_nil("NUMBERP 0", "OUTPUT NUMBERP 0");
}

 void test_list_ops() {
    printf("\n--- List operations ---\n");
    test_cons("FPUT 1 NIL", "OUTPUT FPUT 1 0");
    test_num("FIRST of FPUT", "OUTPUT FIRST FPUT 5 0", 5);
    test_nil("BUTFIRST of FPUT", "OUTPUT BUTFIRST FPUT 5 0");
}

 void test_make_and_vars() {
    printf("\n--- Variables (MAKE / :var) ---\n");
    test_num("MAKE X=42", "MAKE \"X 42 OUTPUT :X", 42);
    test_num("MAKE A=10 B=20", "MAKE \"A 10 MAKE \"B 20 OUTPUT SUM :A :B", 30);
    test_num("MAKE VAL=-5", "MAKE \"VAL -5 OUTPUT ABS :VAL", 5);
}

 void test_if_ifelse() {
    printf("\n--- IF / IFELSE ---\n");
    test_num("IF true", "IF 1 [ MAKE \"R 99 ] OUTPUT :R", 99);
    test_num("IF false", "MAKE \"R 0 IF 0 [ MAKE \"R 99 ] OUTPUT :R", 0);
    test_num("IFELSE true", "IFELSE 1 [ MAKE \"R 10 ] [ MAKE \"R 20 ] OUTPUT :R", 10);
    test_num("IFELSE false", "IFELSE 0 [ MAKE \"R 10 ] [ MAKE \"R 20 ] OUTPUT :R", 20);
}

 void test_repeat() {
    printf("\n--- REPEAT ---\n");
    test_num("REPEAT 5", "MAKE \"C 0 REPEAT 5 [ MAKE \"C SUM :C 1 ] OUTPUT :C", 5);
    test_num("REPEAT 3 add10", "MAKE \"S 0 REPEAT 3 [ MAKE \"S SUM :S 10 ] OUTPUT :S", 30);
    test_num("REPEAT 0", "MAKE \"S 0 REPEAT 0 [ MAKE \"S 99 ] OUTPUT :S", 0);
    test_num("REPEAT -1", "MAKE \"S 0 REPEAT -1 [ MAKE \"S 99 ] OUTPUT :S", 0);
}

 void test_stop() {
    printf("\n--- STOP ---\n");
    test_num("STOP prevents later MAKE", "MAKE \"X 1 STOP MAKE \"X 99 OUTPUT :X", 1);
}

 void test_parens() {
    printf("\n--- Parenthesized expressions ---\n");
    test_num("( SUM 2 3 )", "OUTPUT ( SUM 2 3 )", 5);
    test_num("( 10 + 20 )", "OUTPUT ( 10 + 20 )", 30);
}

 void test_quoted_atoms() {
    printf("\n--- Quoted atoms ---\n");
    test_atom("hello", "OUTPUT \"hello");
    test_atom("cat", "OUTPUT \"cat");
    test_atom("dog", "OUTPUT \"dog");
}

 void test_equalp_atoms() {
    printf("\n--- EQUALP on atoms ---\n");
    test_true("EQUALP cat cat", "OUTPUT EQUALP \"cat \"cat");
    test_nil("EQUALP cat dog", "OUTPUT EQUALP \"cat \"dog");
}

 void test_def_fn() {
    printf("\n--- DEF_FN / TO / OUTPUT ---\n");
    test_num("DEF_FN add1", "DEF_FN \"add1 [:x] [ OUTPUT SUM :x 1 ] OUTPUT add1 5", 6);
    test_num("DEF_FN double", "DEF_FN \"double [:n] [ OUTPUT PRODUCT :n 2 ] OUTPUT double 7", 14);
    test_num("DEF_FN square", "DEF_FN \"square [:x] [ OUTPUT POWER :x 2 ] OUTPUT square 9", 81);
    test_num("TO triple", "TO \"triple [:x] [ OUTPUT PRODUCT :x 3 ] OUTPUT triple 4", 12);
    test_num("DEF_FN negate", "DEF_FN \"negate [:x] [ RETURN MINUS :x ] OUTPUT negate 7", -7);
}

 void test_run_eval() {
    printf("\n--- RUN / EVAL ---\n");
    test_num("RUN block", "MAKE \"X 10 RUN [ MAKE \"X 20 ] OUTPUT :X", 20);
}

 void test_nested_control() {
    printf("\n--- Nested control flow ---\n");
    test_num("IFELSE+REPEAT", "IFELSE 1 [ MAKE \"A 0 REPEAT 3 [ MAKE \"A SUM :A 5 ] ] [ MAKE \"A -1 ] OUTPUT :A", 15);
}

 void test_complex() {
    printf("\n--- Complex expressions ---\n");
    test_num("SUM(PRODUCT)(DIFF)", "OUTPUT SUM PRODUCT 3 4 DIFF 20 8", 24);
    test_num("sequential MAKE", "MAKE \"A 1 MAKE \"A SUM :A 2 MAKE \"A SUM :A 3 OUTPUT :A", 6);
}

 void test_negative_numbers() {
    printf("\n--- Negative numbers ---\n");
    test_num("literal -1", "OUTPUT -1", -1);
    test_num("literal -100", "OUTPUT -100", -100);
    test_num("SUM -5 10", "OUTPUT SUM -5 10", 5);
    test_num("PRODUCT -3 -7", "OUTPUT PRODUCT -3 -7", 21);
    test_num("ABS -999", "OUTPUT ABS -999", 999);
}

int main(int argc, char** argv) {
    install_veh();
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <dll_path>\n", argv[0]);
        return 1;
    }

    v7_lib lib{};
    g_lib = &lib;
    if (v7_load(argv[1], &lib) != 0) {
        fprintf(stderr, "Cannot load %s\n", argv[1]);
        return 1;
    }
    if (!lib.has_v7s_sectorlogo_interp) {
        fprintf(stderr, "DLL missing v7s_sectorlogo_interp\n");
        v7_unload(&lib);
        return 1;
    }

    printf("=== SectorLogo64 Comprehensive Validator ===\n");

    test_integer_literals();
    test_prefix_arithmetic();
    test_infix_arithmetic();
    test_divmod_edge();
    test_prefix_bitwise();
    test_infix_bitwise();
    test_prefix_relational();
    test_infix_relational();
    test_boolean();
    test_list_ops();
    test_make_and_vars();
    test_if_ifelse();
    test_repeat();
    test_stop();
    test_parens();
    test_quoted_atoms();
    test_equalp_atoms();
    test_def_fn();
    test_run_eval();
    test_nested_control();
    test_complex();
    test_negative_numbers();

    v7_unload(&lib);
    g_lib = nullptr;

    printf("\n=== SectorLogo64 Validator: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
"""

with open(out, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Written {len(content)} bytes to {out}")
