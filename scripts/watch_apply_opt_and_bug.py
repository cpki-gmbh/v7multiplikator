import os
import sys
import time

WATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_and_bug")
POLL_SECONDS = 60
BATCH_RETURN_COUNT = 10  # return only when at least this many revised reports are pending


def revised_reports():
	names = []
	for name in sorted(os.listdir(WATCH_DIR)):
		if name.endswith("_revised.md"):
			names.append(name)
	return names


def main():
	while True:
		pending = revised_reports()
		if len(pending) >= BATCH_RETURN_COUNT:
			print("")
			print("..................")
			print("urgent: revised reports ready to apply in opt_and_bug")
			print("")
			print("files to process (ALL of these -- this is not optional):")
			for name in pending:
				print("  " + name)
			print("")
			print("for each of these reports, right now, in the same session:")
			print("  1. read the report in full.")
			print("  2. apply every optimization -- every cpu cycle counts, none may be skipped.")
			print("  3. fix every real bug the report documents -- no exceptions, no deferrals.")
			print("  4. reject only items proven unsafe or impossible (Sec.129a danger-area")
			print("     optimizations, Sec.124 impossible states) -- state the proof.")
			print("")
			print("hard constraints on everything you apply:")
			print("  - applied reports may NEVER hurt performance.")
			print("  - applied reports may NEVER reduce numerical precision (Sec.126 rule 9).")
			print("  - applied reports may NEVER induce bugs -- verify equivalence before editing.")
			print("")
			print("per report, the moment it is consumed (all items applied or rejected")
			print("with proof): DELETE that one .md file IMMEDIATELY -- one del per file,")
			print("BEFORE moving to the next report. NEVER bulk-delete .md files")
			print("(never 'del a.md b.md', never wildcards) -- bulk deletion causes mistakes.")
			print("a still-present .md gets reviewed again, producing duplicate work.")
			print("")
			print("after applying all reports:")
			print("  1. build.bat -- comment checker must show ZERO violations.")
			print("  2. verify.bat SOLO -- all validators + both inference smokes must pass.")
			print("  3. vision smoke when ViT/vision files were touched (baseline must match).")
			print("  4. commit the changes (specific files only, never git add -A).")
			print("     (consumed reports were already deleted one by one at consumption time.)")
			print("  5. run this script again to check for new work. the loop never ends.")
			print("")
			print("this must be done and never avoided.")
			print("..................")
			sys.stdout.flush()
			return
		time.sleep(POLL_SECONDS)


if __name__ == "__main__":
	main()
