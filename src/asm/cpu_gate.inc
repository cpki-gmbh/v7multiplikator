; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define CPUID_LEAF_FEATURES 1                  ; leaf 1: feature bits in ECX
%define CPUID_LEAF_EXTFEAT 7                   ; leaf 7 subleaf 0: extended feature bits in EBX
%define CPUID_MAXBASIC_REQUIRED 7              ; leaf 7 must exist: BMI1/AVX2/BMI2 available there
%define CPUID1_ECX_XSAVE 0x04000000            ; bit 26, xgetbv itself is legal
%define CPUID1_ECX_OSXSAVE 0x08000000          ; bit 27, the OS enabled xgetbv
%define CPUID7_EBX_BMI1 0x00000008             ; bit 3, blsi/blsr/tzcnt, SMT placement, code search
%define CPUID7_EBX_AVX2 0x00000020             ; bit 5, vpermq/vpbroadcastb, all 256-bit kernels
%define CPUID_LEAF_EXT_MAX 0x80000000          ; extended leaves start: EAX = highest extended leaf
%define XCR0_SSE_YMM_MASK 0x00000006           ; bits 1|2: SSE + AVX, YMM state enabled by the OS
%define CPU_SUPPORT_YES 1                      ; published cpu_features_ok when every bit is proven
%define CPU_SUPPORT_NO 0                       ; published cpu_features_ok otherwise, pessimistic default
     section .rdata
cpu_msg_no_support:
     db "[FATAL] cpu_gate: this CPU does not meet the v7 hardware baseline (AVX2, FMA, F16C, BMI1, BMI2, AES-NI, SSE4.1, SSE4.2, RDRAND, RDTSCP, with OS-enabled YMM state)", 0 ; generic fallback published before any test runs
cpu_msg_no_ext_leaf:
     db "[FATAL] cpu_gate: CPUID leaf 7 is unavailable, this CPU predates the AVX2 generation. v7 requires AVX2, FMA, BMI1, BMI2, AES-NI, SSE4.1, SSE4.2 and RDRAND", 0 ; no BMI1/AVX2/BMI2 bits at all
cpu_msg_no_sse41:
     db "[FATAL] cpu_gate: this CPU lacks SSE4.1 (CPUID.01H:ECX[19] = 0), v7's microdb key hash uses vptest", 0 ; first missing silicon feature
cpu_msg_no_sse42:
     db "[FATAL] cpu_gate: this CPU lacks SSE4.2 (CPUID.01H:ECX[20] = 0), v7's microdb key hash uses the crc32 instruction", 0 ; second missing silicon feature
cpu_msg_no_aesni:
     db "[FATAL] cpu_gate: this CPU lacks AES-NI (CPUID.01H:ECX[25] = 0), v7's SectorLogo Intern hash and the sampler RNG use aesenc", 0 ; aes_string_hash / sampler RNG
cpu_msg_no_fma:
     db "[FATAL] cpu_gate: this CPU lacks FMA (CPUID.01H:ECX[12] = 0), every v7 kernel is built on vfmadd/vfnmadd", 0 ; FMA fused ops in every kernel
cpu_msg_no_f16c:
     db "[FATAL] cpu_gate: this CPU lacks F16C, required for FP16 key/value cache conversions", 0 ; vcvtps2ph / vcvtph2ps in the KV cache path
cpu_msg_no_rdrand:
     db "[FATAL] cpu_gate: this CPU lacks RDRAND (CPUID.01H:ECX[30] = 0), the C dispatch autotuner draws its mutation randomness via _rdrand32_step", 0 ; autotune_dispatch.c rng_u32
cpu_msg_no_bmi1:
     db "[FATAL] cpu_gate: this CPU lacks BMI1 (CPUID.07H:EBX[3] = 0), v7's SMT placement uses blsi/blsr and the code search uses tzcnt", 0 ; blsi/blsr/tzcnt sites
cpu_msg_no_avx2:
     db "[FATAL] cpu_gate: this CPU lacks AVX2 (CPUID.07H:EBX[5] = 0), v7's 256-bit kernels use vpermq and AVX2 integer broadcast", 0 ; every 256-bit integer kernel
cpu_msg_no_bmi2:
     db "[FATAL] cpu_gate: this CPU lacks BMI2 (CPUID.07H:EBX[8] = 0), v7's Logo string-search candidate-mask step uses bzhi", 0 ; last silicon feature before the OS-state gate
cpu_msg_no_rdtscp:
     db "[FATAL] cpu_gate: this CPU lacks RDTSCP (CPUID.80000001H:EDX[27] = 0), v7's in-system A/B tuner timing edge uses rdtscp; some hypervisors mask it", 0 ; tuner timing serialization
cpu_msg_no_ymm_state:
     db "[FATAL] cpu_gate: the OS does not enable AVX YMM state (CPUID.01H:ECX[27] OSXSAVE = 0, or XCR0 bits 1|2 clear), every v7 kernel is 256-bit", 0 ; XCR0 OS-state gate failure
; --- CPU_FAIL msg, publish the per-feature message, then return to DllMain ---

%macro CPU_FAIL 1
     lea rax, [rel %1]                         ; rax = the per-feature message to publish
     jmp .cg_fail                              ; jump to the shared publish+return epilogue
%endmacro
     section .text
cpu_gate_probe:
     push rbx                                  ; cpuid writes ebx; rbx is non-volatile
     mov dword [rel cpu_features_ok], CPU_SUPPORT_NO ; pessimistic default: support must be proven, never assumed
     lea rax, [rel cpu_msg_no_support]         ; generic message published BEFORE any test can fail
     mov [rel cpu_gate_msg_ptr], rax           ; so a hand-forced flag still yields a valid string
     xor eax, eax                              ; leaf 0: how many basic leaves exist
     xor ecx, ecx                              ; subleaf 0
     cpuid                                     ; eax = max basic leaf; ebx/ecx/edx = vendor string, unused
     cmp eax, CPUID_MAXBASIC_REQUIRED          ; leaf 7 present?, BMI1/AVX2/BMI2 available there
     jb .no_ext_leaf                           ; no, CPU predates the AVX2 generation
     mov eax, CPUID_LEAF_FEATURES              ; leaf 1: SSE4.1/4.2, FMA, AES-NI, RDRAND, XSAVE, OSXSAVE
     xor ecx, ecx                              ; subleaf 0
     cpuid                                     ; ecx = feature bits, edx = second feature-flag dword, unused
     bt ecx, 19                                ; SSE4.1, vptest in the microdb key hash
     jnc .no_sse41                             ; bit 19 clear, vptest would #UD
     bt ecx, 20                                ; SSE4.2, crc32 in the microdb key hash
     jnc .no_sse42                             ; bit 20 clear, crc32 would #UD
     bt ecx, 25                                ; AES-NI, aesenc in Intern hash and sampler RNG
     jnc .no_aesni                             ; bit 25 clear, aesenc would #UD
     bt ecx, 12                                ; FMA, vfmadd/vfnmadd in every kernel
     jnc .no_fma                               ; bit 12 clear, vfmadd would #UD
     bt ecx, 29                                ; packed FP16 cache converters
     jnc .no_f16c                              ; bit 29 clear, vcvtps2ph would #UD
     bt ecx, 30                                ; RDRAND, the C autotuner's hardware RNG
     jnc .no_rdrand                            ; bit 30 clear, rdrand absent
; --- OS-enabling check: without XCR0[2:1] every VEX.256 raises #UD even though
; the silicon reports AVX2, a hypervisor that does not pass XCR0 through ---
     not ecx                                   ; invert so a cleared OS-state bit shows as set
     test ecx, (CPUID1_ECX_XSAVE | CPUID1_ECX_OSXSAVE) ; both XSAVE and OSXSAVE must be set for xgetbv to be legal
     jne .no_ymm_state                         ; one or both clear, YMM state cannot be assumed
     xor ecx, ecx                              ; xgetbv selector 0 = XCR0
     xgetbv                                    ; eax = XCR0 low, edx = XCR0 high
     not eax                                   ; invert so a cleared state bit shows as set
     test al, XCR0_SSE_YMM_MASK                ; both XCR0 SSE + YMM state bits must be set
     jne .no_ymm_state                         ; one of the two state bits clear, ymm would #UD
     mov eax, CPUID_LEAF_EXTFEAT               ; leaf 7 subleaf 0: BMI1, AVX2, BMI2 in EBX
     xor ecx, ecx                              ; subleaf 0
     cpuid                                     ; ebx = extended feature bits
     test bl, CPUID7_EBX_BMI1                  ; BMI1, blsi/blsr/tzcnt
     jz .no_bmi1                               ; bit 3 clear, blsi/blsr/tzcnt would #UD
     test bl, CPUID7_EBX_AVX2                  ; AVX2, vpermq, vpbroadcastb, every 256-bit kernel
     jz .no_avx2                               ; bit 5 clear, vpermq/AVX2 int would #UD
     bt ebx, 8                                 ; BMI2, bzhi
     jnc .no_bmi2                              ; bit 8 clear, bzhi would #UD
     mov eax, CPUID_LEAF_EXT_MAX               ; leaf 0x80000000: highest extended leaf
     xor ecx, ecx                              ; subleaf 0, ignored for extended leaves, kept tidy
     cpuid                                     ; eax = max extended leaf, >= 0x80000001 on every x64 CPU
     cmp eax, 0x80000001                       ; the RDTSCP leaf itself present?
     jb .no_rdtscp                             ; no, ancient or heavily-masked environment
     mov eax, 0x80000001                       ; leaf 0x80000001: RDTSCP availability in EDX
     xor ecx, ecx                              ; subleaf 0
     cpuid                                     ; edx = extended-processor feature bits
     bt edx, 27                                ; RDTSCP, the tuner's serialized timing edge
     jnc .no_rdtscp                            ; bit 27 clear, rdtscp would #UD, some hypervisors mask it
     mov dword [rel cpu_features_ok], CPU_SUPPORT_YES ; every required bit proven present
     pop rbx                                   ; restore the non-volatile cpuid overwritten
     ret                                       ; return "supported" to DllMain
.no_ext_leaf:
     CPU_FAIL cpu_msg_no_ext_leaf              ; publish the leaf-7-unavailable message
.no_sse41:
     CPU_FAIL cpu_msg_no_sse41                 ; publish the SSE4.1 message
.no_sse42:
     CPU_FAIL cpu_msg_no_sse42                 ; publish the SSE4.2 message
.no_aesni:
     CPU_FAIL cpu_msg_no_aesni                 ; publish the AES-NI message
.no_fma:
     CPU_FAIL cpu_msg_no_fma                   ; publish the FMA message
.no_f16c:
     CPU_FAIL cpu_msg_no_f16c                  ; publish the F16C message
.no_rdrand:
     CPU_FAIL cpu_msg_no_rdrand                ; publish the RDRAND message
.no_bmi1:
     CPU_FAIL cpu_msg_no_bmi1                  ; publish the BMI1 message
.no_avx2:
     CPU_FAIL cpu_msg_no_avx2                  ; publish the AVX2 message
.no_bmi2:
     CPU_FAIL cpu_msg_no_bmi2                  ; publish the BMI2 message
.no_rdtscp:
     CPU_FAIL cpu_msg_no_rdtscp                ; publish the RDTSCP message
.no_ymm_state:
     CPU_FAIL cpu_msg_no_ymm_state             ; publish the OS-state message
.cg_fail:
     mov [rel cpu_gate_msg_ptr], rax           ; publish the message for cpu_gate_fatal
     pop rbx                                   ; restore the non-volatile cpuid overwritten
     ret                                       ; back to DllMain, the public tier terminates
cpu_gate_fatal:
     and rsp, -16                              ; round RSP DOWN to a 16-byte boundary: one instruction, no branch, correct from either parity, RAW entries arrive 8 mod 16, SF-framed entries 0 mod 16 and from any other parity too. Drops RSP by at most 8 bytes, a smaller below-RSP excursion than the 32/40 a two-path probe needs. No matching add rsp: this path never returns, and handle_fatal re-anchors RSP from engine_saved_rsp
     mov rcx, [rel cpu_gate_msg_ptr]           ; arg1 = the message the probe published
     call handle_fatal                         ; print + catch-and-unwind, or terminate, never returns

%macro PUB_CPU_GATE 0
     cmp dword [rel cpu_features_ok], CPU_SUPPORT_YES ; hardware baseline satisfied at load?
     jne cpu_gate_fatal                        ; no, name the missing feature and terminate, never returns
%endmacro
%unmacro CPU_FAIL 1                            ; per-feature message publisher macro, last use above
%undef CPU_SUPPORT_NO
%undef XCR0_SSE_YMM_MASK
%undef CPUID_LEAF_EXT_MAX
%undef CPUID7_EBX_AVX2
%undef CPUID7_EBX_BMI1
%undef CPUID1_ECX_OSXSAVE
%undef CPUID1_ECX_XSAVE
%undef CPUID_MAXBASIC_REQUIRED
%undef CPUID_LEAF_EXTFEAT
%undef CPUID_LEAF_FEATURES
%undef CPUID_LEAF_MAXBASIC
