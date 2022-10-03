import igl
import meshplot as mp
import numpy as np
import pandas as pd

def save(v):
    mp.offline()
    pl = mp.plot(v)#, return_plot=True)
    if pl is not None:
        pl.save("ola.html")

def doplot():
    V = np.genfromtxt("ola.csv", delimiter=",")
    np.isnan(V).any()
    save(V[1:900])
