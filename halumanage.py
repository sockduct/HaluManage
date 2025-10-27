#!/usr/bin/env python3
"""
HaluManage - OpenAI API compatible optimizing inference proxy

This is a thin wrapper that imports and runs the main server from the halumanage package.
For backwards compatibility with direct execution of halumanage.py.
"""

from halumanage import main
from halumanage.server import main

if __name__ == "__main__":
    main()
