#from libplot import doplot; doplot()

import libhack as lh
files = lh.getxmlfiles()
raw = lh.getrawdatafromxml(skip_previoussaved=True)
r = next(raw)
lh.save_data(r)
