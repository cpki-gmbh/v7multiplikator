import os
import sys
import time

WATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_and_bug")
POLL_SECONDS = 15


def unrevised_reports():
	names = []
	for name in sorted(os.listdir(WATCH_DIR)):
		if name.endswith(".md") and not name.endswith("_revised.md"):
			names.append(name)
	return names


def main():
	while True:
		pending = unrevised_reports()
		if pending:
			print("")
			print("..................")
			print("urgent: unrevised reports detected in opt_and_bug")
			for name in pending:
				print("  file to revise: " + name)
			print("")
			print("every claim in these reports must be examined in depth against the source code.")
			print("this must be done and never avoided.")
			print("every claim must be checked if it can be corrected.")
			print("if a claim is not correct, it must be corrected with 200% motivation shown.")
			print("we want as many correct claims as possible to be verified.")
			print("only actionable stuff may be included in the revised report.")
			print("")
			print("after revising all files, start this program again WITHOUT timeout")
			print("(so it waits forever and returns control to you when new reports arrive).")
			print("..................")
			sys.stdout.flush()
			return
		time.sleep(POLL_SECONDS)


if __name__ == "__main__":
	main()
