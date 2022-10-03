import numpy as np
import igl
import meshplot as mp
import time
from datetime import datetime

import libhack as lh

n = 1000
S = None

raw = lh.getrawdatafromxml(skip_previoussaved=True)
try:
    for i, r in enumerate(raw):
        lh.save_data(r)
        s = lh.Surface.fromraw(r)
        if s is None:
            continue

        if S is None:
            S = s
        else:
            S = S.stitch(s)

        submesh = "subsurface_{}_{}_{}.obj".format(lh.ts(), i, n)
        lh.savesurface(s, submesh)
        lh.plotsurface(S, filename="tmpplot.html")

    if S is None: raise Exception("no data")

    mesh = "surfaceplot_{}_{}_{}_{}.obj".format(lh.ts(), n, *S.shape)
    lh.savesurface(S, mesh)
    lh.plotsurface(S)
    lh.Util.serve()

except KeyboardInterrupt:
    print("Exiting")
