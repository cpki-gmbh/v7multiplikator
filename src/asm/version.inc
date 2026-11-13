; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Single source of truth for the release version. check_release.py checks
; this literal against README.md, CHANGELOG.md's top entry and FILE_ID.DIZ
; check9, and v7_get_version hands the pointer to tools and bug reports.
     section .rdata
v7_version_str:
     db "2.36.8", 0                            ; NUL-terminated release version
     section .text
     global v7_get_version                     ; export: release version query
v7_get_version:
     lea rax, [rel v7_version_str]             ; return pointer to the version string
     ret                                       ; plain leaf return, no frame needed
