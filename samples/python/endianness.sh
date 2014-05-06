#!/usr/bin/env python
from struct import pack
if pack('@h', 1) == pack('<h', 1):
    print "Little Endian"
else:
    print "Big Endian"
