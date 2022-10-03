import numpy as np
import igl
import meshplot as mp
import time
from datetime import datetime

import libhack as lh

num_files = lh.countfiles()

n = 10000

S = None
for i in range(1,num_files+1):
    print("loading file %d of %d" % (i, num_files))
    s = lh.Surface.load(i, n=n)
    print("loaded surface. Stitching...")

    S = s if S is None else S.stitch(s)

    print("done.")

    if i % 10 == 0:
        submesh = "stitchedsurface2_{}_{}_{}.obj".format(lh.ts(), i, n)
        lh.savesurface(S, submesh)
        lh.plotsurface(S, filename=submesh+".html")

print("saving mesh")
mesh = "stitchedsurface2_{}_{}.obj".format(lh.ts(), n)
lh.savesurface(S, mesh)
print("done")
