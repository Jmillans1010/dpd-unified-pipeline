"""Create reproducible synthetic data; these are NOT Spectre/PA measurements."""
from pathlib import Path
import argparse
import json
import numpy as np


def create_demo(destination, symbols=160):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError("Demo destination must be new")
    for role,seed in (("training",104), ("validation",209)):
        folder=destination/role; folder.mkdir(parents=True)
        rng=np.random.default_rng(seed)
        qam=(rng.choice([-3,-1,1,3],symbols)+1j*rng.choice([-3,-1,1,3],symbols))*.075
        x=np.repeat(qam,16)
        kernel=np.hanning(9); kernel/=kernel.sum()
        x=np.convolve(x,kernel,mode="same")
        y=(1.3+.3j)*(x*(1+.35*abs(x)**2)+.02*np.r_[0j,x[:-1]])
        y=np.r_[np.zeros(3,complex),y[:-3]]
        y+=1e-5*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
        time=8e-9+np.arange(len(x))*250e-12
        for ch,values in (("I_in",x.real),("Q_in",x.imag),("I_out",y.real),("Q_out",y.imag)):
            np.savetxt(folder/f"{ch}_{role}.csv",np.column_stack([time,values]),delimiter=",",header=f"/{ch.upper()} X,/{ch.upper()} Y",comments="",fmt="%.17g")
        (folder/"input").write_text('// Synthetic fixture metadata only. Not a runnable PA netlist.\nsimulator lang=spectre\nW0 (I_IN 0 Q_IN 0) wsource standard="MSource" MSource_ModType=QAM16 MSource_DataRate=1G MSource_Frequency=240G MSource_OFDMType=No oversample=4\ntran tran stop=2.2u strobeperiod=250p strobeoutput=strobeonly\n',encoding="utf-8")
    config=json.loads(Path(__file__).with_name("pipeline.json").read_text())
    config.update(training_dir="training",validation_dir="validation",output_dir="runs",
                  seed_provenance={"training":104,"validation":209,"declaration":"Synthetic fixtures; not PA measurements"})
    config["models"]={"MP":{"max_P":5,"max_M":2},"EMP":{"max_P":5,"max_M":2},
                      "GMP":{"max_P":3,"max_M":1,"max_Q":1},"PrunedVolterra":{"max_P":3,"max_M":2,"max_D":1},
                      "LUT":{"min_bins":4,"max_bins":8}}
    config["search"].update(max_candidates=30,max_iterations=20,plateau_extra_depth=1)
    (destination/"demo.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    return destination/"demo.json"


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("destination"); a=p.parse_args()
    print(create_demo(a.destination))
