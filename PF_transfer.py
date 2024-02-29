import py_dss_interface
import pathlib
import os
from Methods import SmartInverterFunction
import pandas as pd
import numpy as np
from data_process import strfill


# Set the parameters relevant to the power flow calculation.
def set_baseline():
    dss.text("New Energymeter.m1 Line.ln5815900-1 1")
    dss.text("Set Maxiterations=100")
    dss.text("set maxcontrolit=100")
    dss.text("set Maxcontroliter=100")
    dss.text("set Maxiter=100")
    dss.text("Batchedit Load..* daily=default")


def set_time_series_simulation():
    dss.text("set controlmode=Static")
    dss.text("set mode=Snap")
    # dss.text("set number=24")
    # dss.text("set stepsize=1h")


def get_energymeter_results():
    dss.meters_write_name("m1")
    feeder_kwh = dss.meters_register_values()[0]
    feeder_kvarh = dss.meters_register_values()[1]
    loads_kwh = dss.meters_register_values()[4]
    losses_kwh = dss.meters_register_values()[12]
    pv_kwh = loads_kwh + losses_kwh - feeder_kwh

    return feeder_kwh, feeder_kvarh, loads_kwh, losses_kwh, pv_kwh


def generate_opendss_files(YS_out, columns):
    v = YS_out.values[:, columns]  # Select the designated column data.
    v = np.around(v, 2)  # Retain two decimal places.

    with open('./Feeders/8500-Node/Loads.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
    head_str = str_temp[:12]
    str_temp = str_temp[12:1189]
    str(v[0])
    type(strfill(str(v[0]), 7, " "))

    for col, col_idx in enumerate(columns):
        with open(f'./Feeders/8500-node/Loads_test{col+1}.dss', 'w+') as f:
            for str_ in head_str:
                f.write(str_ + "\n")
            for i in range(len(str_temp)):
                str_ = str_temp[i]
                v_ = v[i, col]
                str_ = str_[:98] + str(v_).ljust(7, " ") + str_[98 + 7:]
                f.write(str_ + "\n")

YS_out = pd.read_csv('./transfer/MPGTN_YS_pre_out.csv', header=None)
# YS_out = np.load('MPGTN_prediction.npy')
# v = YS_out[:, 2]
columns = [0, 2, 5]  # Specify the column indices to be selected.
generate_opendss_files(YS_out, columns)

smart_inverter_functions_list = ["unity-pf", "pf", "volt-var"]
#Read the parameter file.
script_path = os.path.dirname(os.path.abspath(__file__))
dss_file = pathlib.Path(script_path).joinpath("Feeders", "8500-Node", "Master.dss")  #通过master文件去选择参数文件

dss = py_dss_interface.DSSDLL()

bus = "l3104830"

feeder_kwh_list = list()
feeder_kvarh_list = list()
loads_kwh_list = list()
losses_kwh_list = list()
pv_kwh_list = list()

for smart_inverter_function in smart_inverter_functions_list:
    # Process for each smart inverter function
    dss.text(f"Compile [{dss_file}]")
    set_baseline()
    set_time_series_simulation()

    # Add PV system and the smart inverter function
    SmartInverterFunction(dss, bus, 12.47, 8000, 8000, smart_inverter_function)

    dss.text("solve")

    # Read Energymeter results
    energymeter_results = get_energymeter_results()
    feeder_kwh_list.append(energymeter_results[0])
    feeder_kvarh_list.append(energymeter_results[1])
    loads_kwh_list.append(energymeter_results[2])
    losses_kwh_list.append(energymeter_results[3])
    pv_kwh_list.append(energymeter_results[4])

# dss.text("Show Eventlog") #View the event log according to the 'Show' command in the manual document.

# Save results in a csv file
dict_to_df = dict()
dict_to_df["smart_inverter_function"] = smart_inverter_functions_list
dict_to_df["feeder_kwh"] = feeder_kwh_list
dict_to_df["feeder_kvarh"] = feeder_kvarh_list
dict_to_df["loads_kwh"] = loads_kwh_list
dict_to_df["losses_kwh"] = losses_kwh_list
dict_to_df["pv_kwh"] = pv_kwh_list

df = pd.DataFrame().from_dict(dict_to_df)


dss.text("Show voltages LL nodes")  #Generate the result in a txt file.
dss.text("Show voltages LN nodes")
dss.text("Export Voltages")

# dss.text("Plot Profile  Phases=ALL")  #
# dss.text("Plot type=circuit quantity=power")
# dss.text("Plot Circuit Losses 1phlinestyle=3")
# dss.text("Plot Circuit quantity=3 object=mybranchdata.csv")
# dss.text("Plot General quantity=1 object=mybusdata.csv")
# dss.text("Plot profile phases=Angle")
dss.text("Plot profile phases=LL3ph")
dss.text("Plot profile phases=LLall")
dss.text("Plot Profile Phases=Primary")
# dss.text("plot circuit Power Max=2000 dots=n labels=n subs=n C1=$00FF0000")


output_file = pathlib.Path(script_path).joinpath("outputs", "results.csv")
df.to_csv(output_file, index=False)