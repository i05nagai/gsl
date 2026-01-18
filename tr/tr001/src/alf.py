#!/usr/bin/env python3
#
# Generate Schmidt normalized ALFs up to a given order;
# this script is superseded by the Mathematica notebook files,
# which additionally compute 1st and 2nd ALF derivatives

from mpmath import mp

def schmidt_norm(l, m):
  if m == 0:
    return mp.mpf(1)
  elif m % 2 != 0:
    return mp.mpf(-1) * mp.sqrt(2 * mp.fac(l-m) / mp.fac(l+m))
  else:
    return mp.sqrt(2 * mp.fac(l-m) / mp.fac(l+m))

def Slm(l,m,x):
  return schmidt_norm(l,m) * mp.legenp(l,m,x)

mp.dps = 25
mp.pretty = True

L = 3
M = L
x = mp.mpf(0.15)

print("{ ", end='')

for m in range(M+1):
  for l in range(m,L+1):
    print(mp.nstr(Slm(l,m,x), 20, strip_zeros=False),", ", end='')

print("};")
