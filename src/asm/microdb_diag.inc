; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
db_set_readonly:
     mov dword [rel mdb_readonly], ecx         ; publish the readonly flag, nonzero makes db_save a no-op, persistence gate only; in-memory ht_add keeps working so RAG DatabaseSet stays live, DWORD width is load-bearing: g_chat_eos_token, dd -1, the first-run EOS sentinel is the immediately adjacent datum, v7_data.inc, no padding ; a qword store would zero it with rcx's upper half
     ret                                       ; back to caller
