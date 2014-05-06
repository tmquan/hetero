#!/usr/bin/env python
from sys import argv
from array import array

a = array("H", open(argv[1], "rb").read())
a.byteswap()
open(argv[2], "wb").write(a.tostring())

