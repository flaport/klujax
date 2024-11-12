import glob
import os
import sys
from subprocess import call

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_PATH = os.path.join(PROJECT, "tests.py")

print(f"{PROJECT=}", os.path.exists(PROJECT))
print(f"{TESTS_PATH=}", os.path.exists(PROJECT))

if sys.platform == "linux":
    exit(call(["pytest", "-s", TESTS_PATH]))
elif sys.platform == "win32":
    exit(call(["pytest", "-s", TESTS_PATH]))
elif sys.platform == "darwin":
    wheelhouse = "/Users/runner/work/klujax/klujax/wheelhouse"
    print(f"{wheelhouse=}", os.path.isdir(wheelhouse))
    print(os.listdir(wheelhouse))
    wheelpaths = glob.glob(f"{wheelhouse}/*arm*")
    print(f"{wheelpaths=}")
    wheelpath = glob.glob(f"{wheelhouse}/*arm*")[0]
    print(f"{wheelpath=}")
    print(f"pip install {wheelpath}")
    (code := call(["pip", "install", wheelpath])) and exit(code)
    exit(call(["pytest", "-s", TESTS_PATH]))
