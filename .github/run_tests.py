import os
import platform
import sys
from subprocess import call

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_PATH = os.path.join(PROJECT, "tests.py")

print(f"{PROJECT=}", os.path.exists(PROJECT))
print(f"{TESTS_PATH=}", os.path.exists(PROJECT))

if sys.platform == "darwin":
    architecture = platform.machine()
    print(f"{architecture=}")
    if architecture != "arm64":
        exit(print("skipping tests as we only run them on arm64."))

exit(call(["pytest", "-s", TESTS_PATH]))
