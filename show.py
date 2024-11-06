import os

os.environ["OMP_NUM_THREADS"] = "1"

from MVHSC_Aux import create_instances


settings = {}
DI, IN, CL, EV, IT = create_instances(settings,0)

data = EV.use_result({}, "load")
EV.plot_result(data, ["nmi","grad","val"])


print("aa")