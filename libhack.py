import sys
import igl
import http.server
import socketserver
import numpy as np
import pandas as pd
import meshplot as mp
import time
from datetime import datetime
from dataclasses import dataclass
sys.path.extend([r'./opt/'])
import pds4_tools

def getfiles():
    return pds4_tools.read("https://sbnarchive.psi.edu/pds4/orex/orex.ola/data_calibrated_v2/collection_ola_data_calibrated_v2.xml")

def getxmlfiles(skip_previoussaved=True):
    import csv
    files = []
    url = "https://sbnarchive.psi.edu/pds4/orex/orex.ola/data_calibrated_v2/orbit_b/"
    with open("files.csv") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            f = url + row[1]
            if 'xml' in f: files.append(f)

    count = 0 if not skip_previoussaved else countfiles()
    return files[count:]

def _getrawdatafromxml(files, debug=True):
    for f in files:
        if debug: print(f)
        data = pds4_tools.read(f)
        yield data

def getrawdatafromxml(skip_previoussaved=True):
    files = getxmlfiles(skip_previoussaved)
    yield from _getrawdatafromxml(files)

def tovertices(data):
    table_data = data[0]
    points = np.c_[ table_data['x'], table_data['y'], table_data['z'] ]
    return points

def totimestamps(data):
    table_data = data[0]
    return table_data['utc']

def countfiles():
    import os, os.path
    DIR = "./data"
    return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

def save_data(data):
    import pandas as pd
    v = tovertices(data)
    x, y, z = v[:,0], v[:,1], v[:,2]
    t = totimestamps(data)
    ti = np.c_[ x, y, z ]
    df = pd.DataFrame(ti)
    i = countfiles() + 1
    df.to_csv("./data/"+str(i)+".csv", index=False)
    return data

def open_data(i):
    return np.genfromtxt("./data/"+str(i)+".csv", delimiter=",")

def getmydatacloud(i=3):
    v = open_data(i)
    return v[1:]

def sample(v, n = 5000):
    return v[np.random.choice(v.shape[0], n)]

def boundingbox(v):
    return igl.bounding_box(v)

def maxdistance(v):
    return igl.all_pairs_distances(v, v, True).max()

def partition(v, groups=2):
    G, *_ = igl.partition(v, groups)
    return v[G==1]

def cleanpointcloud(v, n=5000, partition_vertices=2):
    # TODO: remove outliers
    return sample(partition(v, partition_vertices) if partition_vertices > 1 else v, n=n)

def mappointcloud(raw, n=5000):
    v = tovertices(raw)
    V = cleanpointcloud(v, n=n)
    return V, get_faces(V)

def fixuppointcloud(i=2,n=5000):
    return cleanpointcloud(getmydatacloud(i=i), n=n)

def cleanplot(i, n):
    V = fixuppointcloud(i=i,n=n)
    F = get_faces(V)
    p = _cleanplot(V,F)
    filename = "plot_"+str(i)+"_"+str(n)+".html"
    print("saving to", filename)
    if p is not None:
        p.save(filename)
    else:
        print("no plot")

def _cleanplot(V,F, filename=None, draw_edges=False):
    mp.offline()
    p = mp.plot(V, F, return_plot=True)
    if draw_edges:
        E = igl.edges(F)
        p.add_edges(V, F, shading={"line_color": "red"})

    if filename is not None:
        #filename = "plot_"+str(i)+"_"+str(n)+".html"
        print("saving to", filename)
        if p is not None:
            p.save(filename)
        else:
            print("no plot")
        return p

def get_faces(v):
    vo = np.zeros((v.shape[0], 2))
    Vo = np.zeros((v.shape[0], 3))

    Vo = project_to_plane(v)
    vo = Vo[:,0:2]

    F = igl.delaunay_triangulation(vo)
    return F

def project_to_plane(V):
    n , c = igl.fit_plane(V)
    Vo = np.zeros((V.shape[0], 3))
    for i in range(V.shape[0]):
        v = V[i]
        Vo[i] = v - np.dot(v - c, n) * n
    return Vo

def try_write_mesh_fn(fn, v, f, DIR="./meshes/"):
    try:
        filename = DIR+fn()
        print("saving mesh to "+ filename)
        igl.write_triangle_mesh(filename, v, f)
        return True
    except:
        print("failed to write mesh to "+ fn)
        return False

def try_write_mesh(filename, v, f):
    return try_write_mesh_fn(lambda: filename, v, f)

def ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

@dataclass
class Surface:
    V: np.ndarray
    F: np.ndarray
    N: np.int64
    M: np.int64

    def __init__(self, V, F):
        self.V = V
        self.F = F
        self.N = V.shape[0]
        self.M = F.shape[0]
        self.shape = (self.N, self.M)
        self.G = (self.V, self.F)
        self.xyz = (self.V[:,0], self.V[:,1], self.V[:,2])

    def __repr__(self):
        return f"Surface(V={self.V.shape}, F={self.F.shape})"

    @staticmethod
    def load(i, n=5000):
        v = fixuppointcloud(i=i,n=n)
        f = get_faces(v)
        return Surface(v, f)

    def plot(self, filename):
        return _cleanplot(self.V, self.F, filename)

    def save(self, filename=None):
        if filename is None:
            filename = "unnamed_mesh_{}_{}_{}_{}.obj".format(ts(), self.N, self.M, *self.shape)
        return try_write_mesh(filename, self.V, self.F)

    @staticmethod
    def combine(s1, s2):
        return s1.stitch(s2)

    def stitch(self, other):
        V = np.concatenate((self.V, other.V))
        F = np.concatenate((self.F, other.F + self.N))
        return Surface(V, F)

    @staticmethod
    def fromraw(raw):
        try:
            v, f = mappointcloud(raw)
            return Surface(v, f)
        except Exception as e:
            print(e)
            return None

def plotsurface(s, filename="plot.html"):
    if s is not None:
        s.plot(filename)
    else:
        raise Exception("no surface to plot")

def savesurface(s, f):
    if s is not None:
        s.save(f)
    else:
        raise Exception("no surface to save")

PORT = 80
class Util:
    @staticmethod
    def serve():
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("serving at port", PORT)
            httpd.serve_forever()

