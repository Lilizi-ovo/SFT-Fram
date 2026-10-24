"""Private Bash-to-Python bridge. Its stdout is captured, never logged."""

import json
import os
import sys


if __name__ == "__main__":
    print(json.dumps({"args": sys.argv[1:], "env": dict(os.environ)}, ensure_ascii=True))
