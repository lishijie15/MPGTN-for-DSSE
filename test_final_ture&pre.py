import py_dss_interface
import pathlib
import os
from Methods import SmartInverterFunction
from data_process import strfill
import pandas as pd
import numpy as np
from metrics import evaluate

def set_baseline():
    smart_inverter_functions_list = ["unity-pf", "pf", "volt-var"]
    # Read the parameter file.
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("Feeders", "8500-Node", "Master.dss")

    dss = py_dss_interface.DSSDLL()


    dss.text("New Energymeter.m1 Line.ln5815900-1 1")
    dss.text("Set Maxiterations=100")
    dss.text("set maxcontrolit=100")
    dss.text("set Maxcontroliter=100")
    dss.text("set Maxiter=100")
    dss.text("Batchedit Load..* daily=default")
def set_time_series_simulation():
    smart_inverter_functions_list = ["unity-pf", "pf", "volt-var"]
    # Read the parameter file.
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("Feeders", "8500-Node", "Master.dss")

    dss = py_dss_interface.DSSDLL()


    dss.text("set controlmode=Static")
    dss.text("set mode=Snap")
    # dss.text("set number=24")
    # dss.text("set stepsize=1h")
def get_energymeter_results():
    smart_inverter_functions_list = ["unity-pf", "pf", "volt-var"]
    # Read the parameter file.
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("Feeders", "8500-Node", "Master.dss")

    dss = py_dss_interface.DSSDLL()


    dss.meters_write_name("m1")
    feeder_kwh = dss.meters_register_values()[0]
    feeder_kvarh = dss.meters_register_values()[1]
    loads_kwh = dss.meters_register_values()[4]
    losses_kwh = dss.meters_register_values()[12]
    pv_kwh = loads_kwh + losses_kwh - feeder_kwh

    return feeder_kwh, feeder_kvarh, loads_kwh, losses_kwh, pv_kwh

def master_1(abs_path):
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
        str_temp[17]="Redirect  Loads_test1.dss     ! unBalanced Loads"
    # 读取 master.dss 文件
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'w+') as f:
        for i in range(len(str_temp)):
            str_ = str_temp[i]
            f.write(str_ + "\n")
def master_2(abs_path):
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
        str_temp[17]="Redirect  Loads_test2.dss     ! unBalanced Loads"
    # 读取 master.dss 文件
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'w+') as f:
        for i in range(len(str_temp)):
            str_ = str_temp[i]
            f.write(str_ + "\n")
def master_3(abs_path):
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
        str_temp[17]="Redirect  Loads_test3.dss     ! unBalanced Loads"
    # 读取 master.dss 文件
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'w+') as f:
        for i in range(len(str_temp)):
            str_ = str_temp[i]
            f.write(str_ + "\n")
def master_4(abs_path):
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
        str_temp[17]="Redirect  Loads_test4.dss     ! unBalanced Loads"
    # 读取 master.dss 文件
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'w+') as f:
        for i in range(len(str_temp)):
            str_ = str_temp[i]
            f.write(str_ + "\n")
def master_5(abs_path):
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
        str_temp[17]="Redirect  Loads_test5.dss     ! unBalanced Loads"
    # 读取 master.dss 文件
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'w+') as f:
        for i in range(len(str_temp)):
            str_ = str_temp[i]
            f.write(str_ + "\n")
def master_6(abs_path):
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
        str_temp[17]="Redirect  Loads_test6.dss     ! unBalanced Loads"
    # 读取 master.dss 文件
    with open(f'{abs_path}/Feeders/8500-Node/Master.dss', 'w+') as f:
        for i in range(len(str_temp)):
            str_ = str_temp[i]
            f.write(str_ + "\n")
def master_time(t, abs_path):
    if t == 0:
        master_1(abs_path)
    elif t == 1:
        master_2(abs_path)
    elif t == 2:
        master_3(abs_path)
    elif t == 3:
        master_4(abs_path)
    elif t == 4:
        master_5(abs_path)
    elif t == 5:
        master_6(abs_path)
    else:
        print("No function defined for time", t)
def run_save():
    smart_inverter_functions_list = ["unity-pf", "pf", "volt-var"]
    # Read the parameter file.
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("Feeders", "8500-Node", "Master.dss")

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

    dss.text("Show voltages LL nodes")  # Generate the result in a txt file.
    dss.text("Show voltages LN nodes")
    dss.text("Export Voltages")
def data_reader_initial(abs_path):
    #with open('./Feeders/8500-node/IEEE8500_VLN_Node.Txt', 'r') as f:
    with open(f'{abs_path}/Feeders/8500-Node/IEEE8500_VLN_Node.Txt', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
        head_str = str_temp[:5]
        V_LN = str_temp[5:8539]
        output_VLN_cur = []
        output_Angle_cur = []
        output_BaseV_cur = []
        for i in range(len(V_LN)):
            str_ = V_LN[i]
            number_VLN_cur = str_[29:35]
            number_Ang_cur = str_[39:45]
            number_base_cur = str_[58:65]
            number_V_cur = float(number_VLN_cur)
            number_A_cur = float(number_Ang_cur)
            number_b_cur = float(number_base_cur)
            output_VLN_cur.append(number_V_cur)
            output_Angle_cur.append(number_A_cur)
            output_BaseV_cur.append(number_b_cur)
        pu_cur = [a / b for a, b in zip(output_VLN_cur, output_BaseV_cur)]
    output_VLN = output_VLN_cur
    output_Angle = output_Angle_cur
    output_BaseV = output_BaseV_cur
    pu = pu_cur
    return output_VLN, output_Angle, output_BaseV, pu
def data_reader_normal(middle_VLN, middle_Angle, middle_BaseV, middle_pu, t, Begin_group, abs_path):
    with open(f'{abs_path}/Feeders/8500-Node/IEEE8500_VLN_Node.Txt', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
        head_str = str_temp[:5]
        V_LN = str_temp[5:8539]
        output_VLN_cur = []
        output_Angle_cur = []
        output_BaseV_cur = []

        for i in range(len(V_LN)):
            str_ = V_LN[i]
            number_VLN_cur = str_[29:35]
            number_Ang_cur = str_[39:45]
            number_base_cur = str_[58:65]
            number_V_cur = float(number_VLN_cur)
            number_A_cur = float(number_Ang_cur)
            number_b_cur = float(number_base_cur)
            output_VLN_cur.append(number_V_cur)
            output_Angle_cur.append(number_A_cur)
            output_BaseV_cur.append(number_b_cur)
        pu_cur = [a / b for a, b in zip(output_VLN_cur, output_BaseV_cur)]

    if t == Begin_group:
        middle_VLN = list(zip(middle_VLN, output_VLN_cur))
        middle_Angle = list(zip(middle_Angle,  output_Angle_cur))
        middle_BaseV = list(zip(middle_BaseV, output_BaseV_cur))
        middle_pu = list(zip(middle_pu, pu_cur))
    elif t == Begin_group + 1:
        middle_VLN = [[row1] + [value] for row1, value in zip(middle_VLN, output_VLN_cur)]
        middle_Angle = [[row1] + [value] for row1, value in zip(middle_Angle, output_Angle_cur)]
        middle_BaseV = [[row1] + [value] for row1, value in zip(middle_BaseV, output_BaseV_cur)]
        middle_pu = [[row1] + [value] for row1, value in zip(middle_pu, pu_cur)]
    else:
        middle_VLN = [row1 + [value] for row1, value in zip(middle_VLN, output_VLN_cur)]
        middle_Angle = [row1 + [value] for row1, value in zip(middle_Angle, output_Angle_cur)]
        middle_BaseV = [row1 + [value] for row1, value in zip(middle_BaseV, output_BaseV_cur)]
        middle_pu = [row1 + [value] for row1, value in zip(middle_pu, pu_cur)]


    return middle_VLN, middle_Angle, middle_BaseV, middle_pu

def generate_opendss_files(YS_out, columns, abs_path):
    v = YS_out[:, columns]  # Select the designated column data.
    v = np.around(v, 2)  # Retain two decimal places.


    with open(f'{abs_path}/Feeders/8500-Node/Loads.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
    head_str = str_temp[:12]
    str_temp = str_temp[12:1189]
    str(v[0])

    type(strfill(str(v[0]), 7, " "))

    for col, col_idx in enumerate(columns):
        with open(f'{abs_path}/Feeders/8500-node/Loads_test{col+1}.dss', 'w+') as f:
            for str_ in head_str:
                f.write(str_ + "\n")
            for i in range(len(str_temp)):
                str_ = str_temp[i]
                v_ = v[i, col]
                str_ = str_[:98] + str(v_).ljust(7, " ") + str_[98 + 7:]
                f.write(str_ + "\n")

task = 'MPGTN'
YS_out_ground = np.load('./data&result/result_npy/MPGTN_PF/MPGTN_groundtruth.npy')
YS_out_pre = np.load('./data&result/result_npy/MPGTN_PF/MPGTN_prediction.npy')

Begin_group = 0
Final_group = 400

abs_path = os.path.dirname(os.path.abspath(__file__))

YS_final_ground = np.transpose(YS_out_ground, (0, 2, 1))
YS_final_pre = np.transpose(YS_out_pre, (0, 2, 1))
columns = [0, 1, 2, 3, 4, 5]

t_1 = Begin_group
while t_1 < Final_group:
    generate_opendss_files(YS_final_pre[t_1], columns, abs_path)
    for t_2 in range(0,6):
        master_time(t_2, abs_path)
        run_save()
        if t_2 == 0:
            if t_1 == Begin_group:
                middle_VLN0, middle_Angle0, middle_BaseV0, middle_pu0 = data_reader_initial(abs_path)
            else:
                middle_VLN0, middle_Angle0, middle_BaseV0, middle_pu0 = data_reader_normal(middle_VLN0, middle_Angle0, middle_BaseV0, middle_pu0, t_1, Begin_group, abs_path)
        if t_2 == 1:
            if t_1 == Begin_group:
                middle_VLN1, middle_Angle1, middle_BaseV1, middle_pu1 = data_reader_initial(abs_path)
            else:
                middle_VLN1, middle_Angle1, middle_BaseV1, middle_pu1 = data_reader_normal(middle_VLN1, middle_Angle1, middle_BaseV1, middle_pu1, t_1, Begin_group, abs_path)
        if t_2 == 2:
            if t_1 == Begin_group:
                middle_VLN2, middle_Angle2, middle_BaseV2, middle_pu2 = data_reader_initial(abs_path)
            else:
                middle_VLN2, middle_Angle2, middle_BaseV2, middle_pu2 = data_reader_normal(middle_VLN2, middle_Angle2, middle_BaseV2, middle_pu2, t_1, Begin_group, abs_path)
        if t_2 == 3:
            if t_1 == Begin_group:
                middle_VLN3, middle_Angle3, middle_BaseV3, middle_pu3 = data_reader_initial(abs_path)
            else:
                middle_VLN3, middle_Angle3, middle_BaseV3, middle_pu3 = data_reader_normal(middle_VLN3, middle_Angle3, middle_BaseV3, middle_pu3, t_1, Begin_group, abs_path)
        if t_2 == 4:
            if t_1 == Begin_group:
                middle_VLN4, middle_Angle4, middle_BaseV4, middle_pu4 = data_reader_initial(abs_path)
            else:
                middle_VLN4, middle_Angle4, middle_BaseV4, middle_pu4 = data_reader_normal(middle_VLN4, middle_Angle4, middle_BaseV4, middle_pu4, t_1, Begin_group, abs_path)
        if t_2 == 5:
            if t_1 == Begin_group:
                middle_VLN5, middle_Angle5, middle_BaseV5, middle_pu5 = data_reader_initial(abs_path)
            else:
                middle_VLN5, middle_Angle5, middle_BaseV5, middle_pu5 = data_reader_normal(middle_VLN5, middle_Angle5, middle_BaseV5, middle_pu5, t_1, Begin_group, abs_path)
        print('t_1=', t_1, '&&t_2=', t_2)
    t_1 += 1
output_Angel_pre = np.array([middle_Angle0, middle_Angle1, middle_Angle2, middle_Angle3, middle_Angle4, middle_Angle5])
output_pu_pre = np.array([middle_pu0, middle_pu1, middle_pu2, middle_pu3, middle_pu4, middle_pu5])#(6, 8534, 3)
output_Angel = np.transpose(output_Angel_pre, (2, 0, 1))
output_pu = np.transpose(output_pu_pre, (2, 0, 1))#(3,6,8534)

#
t_3 = Begin_group
while t_3 < Final_group:
    generate_opendss_files(YS_final_ground[t_3], columns, abs_path)
    for t_4 in range(0,6):
        master_time(t_4, abs_path)
        run_save()
        if t_4 == 0:
            if t_3 == Begin_group:
                middle_VLN_ground0, middle_Angle_ground0, middle_BaseV_ground0, middle_pu_ground0 = data_reader_initial(abs_path)
            else:
                middle_VLN_ground0, middle_Angle_ground0, middle_BaseV_ground0, middle_pu_ground0 = data_reader_normal(middle_VLN_ground0, middle_Angle_ground0, middle_BaseV_ground0, middle_pu_ground0, t_3, Begin_group, abs_path)
        if t_4 == 1:
            if t_3 == Begin_group:
                middle_VLN_ground1, middle_Angle_ground1, middle_BaseV_ground1, middle_pu_ground1 = data_reader_initial(abs_path)
            else:
                middle_VLN_ground1, middle_Angle_ground1, middle_BaseV_ground1, middle_pu_ground1 = data_reader_normal(middle_VLN_ground1, middle_Angle_ground1, middle_BaseV_ground1, middle_pu_ground1, t_3, Begin_group, abs_path)
        if t_4 == 2:
            if t_3 == Begin_group:
                middle_VLN_ground2, middle_Angle_ground2, middle_BaseV_ground2, middle_pu_ground2 = data_reader_initial(abs_path)
            else:
                middle_VLN_ground2, middle_Angle_ground2, middle_BaseV_ground2, middle_pu_ground2 = data_reader_normal(middle_VLN_ground2, middle_Angle_ground2, middle_BaseV_ground2, middle_pu_ground2, t_3, Begin_group, abs_path)
        if t_4 == 3:
            if t_3 == Begin_group:
                middle_VLN_ground3, middle_Angle_ground3, middle_BaseV_ground3, middle_pu_ground3 = data_reader_initial(abs_path)
            else:
                middle_VLN_ground3, middle_Angle_ground3, middle_BaseV_ground3, middle_pu_ground3 = data_reader_normal(middle_VLN_ground3, middle_Angle_ground3, middle_BaseV_ground3, middle_pu_ground3, t_3, Begin_group, abs_path)
        if t_4 == 4:
            if t_3 == Begin_group:
                middle_VLN_ground4, middle_Angle_ground4, middle_BaseV_ground4, middle_pu_ground4 = data_reader_initial(abs_path)
            else:
                middle_VLN_ground4, middle_Angle_ground4, middle_BaseV_ground4, middle_pu_ground4 = data_reader_normal(middle_VLN_ground4, middle_Angle_ground4, middle_BaseV_ground4, middle_pu_ground4, t_3, Begin_group, abs_path)
        if t_4 == 5:
            if t_3 == Begin_group:
                middle_VLN_ground5, middle_Angle_ground5, middle_BaseV_ground5, middle_pu_ground5 = data_reader_initial(abs_path)
            else:
                middle_VLN_ground5, middle_Angle_ground5, middle_BaseV_ground5, middle_pu_ground5 = data_reader_normal(middle_VLN_ground5, middle_Angle_ground5, middle_BaseV_ground5, middle_pu_ground5, t_3, Begin_group, abs_path)
        print('t_3=', t_3, '&&t_4=', t_4)
    t_3 += 1
ground_Angel = np.array([middle_Angle_ground0, middle_Angle_ground1, middle_Angle_ground2, middle_Angle_ground3, middle_Angle_ground4, middle_Angle_ground5])
ground_pu = np.array([middle_pu_ground0, middle_pu_ground1, middle_pu_ground2, middle_pu_ground3, middle_pu_ground4, middle_pu_ground5])#(6, 8534, 3)
ground_Angel = np.transpose(ground_Angel, (2, 0, 1))
ground_pu = np.transpose(ground_pu, (2, 0, 1))#(3,6,8534)

#abnormal
output_pu[output_pu > 100000] = output_pu[output_pu > 100000] / 1000000
ground_pu[ground_pu > 100000] = ground_pu[ground_pu > 100000] / 1000000


with open(f'{abs_path}/data&result/result_txt/{task}_score_{Final_group-Begin_group}.txt', 'w') as f:
#with open(f'{abs_path}/data&result/result_txt/{task}_{Final_group-Begin_group}.txt', 'w') as f:
    f.write("Angel\n")
    MSE_A, RMSE_A, MAE_A, MAPE_A, skewness_A, kurtosis_A = evaluate(ground_Angel, output_Angel)
    f.write(" all pred steps, MSE, RMSE, MAE, MAPE, skewness, kurtosis, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (MSE_A, RMSE_A, MAE_A, MAPE_A, skewness_A, kurtosis_A))
    for i in range(6):
        MSE, RMSE, MAE, MAPE, skewness, kurtosis = evaluate(ground_Angel[:, i, :], output_Angel[:, i, :])
        #MSE, RMSE, MAE, MAPE, skewness, kurtosis = evaluate(Angel_groud[i, :], output_Angel[i, :])
        f.write("%d step,MSE, RMSE, MAE, MAPE, skewness, kurtosis, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (i+1, MSE, RMSE, MAE, MAPE, skewness, kurtosis))
    f.write("\n")
    f.write("\n")

    f.write("p.u\n")
    MSE_p, RMSE_p, MAE_p, MAPE_p, skewness_p, kurtosis_p = evaluate(ground_pu, output_pu)
    f.write(" all pred steps, MSE, RMSE, MAE, MAPE, skewness, kurtosis, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (MSE_p, RMSE_p, MAE_p, MAPE_p, skewness_p, kurtosis_p))
    for i in range(6):
        MSE, RMSE, MAE, MAPE, skewness, kurtosis = evaluate(ground_pu[:, i, :], output_pu[:, i, :])
        #MSE, RMSE, MAE, MAPE, skewness, kurtosis = evaluate(Angel_groud[i, :], output_Angel[i, :])
        f.write("%d step,MSE, RMSE, MAE, MAPE, skewness, kurtosis, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (i+1, MSE, RMSE, MAE, MAPE, skewness, kurtosis))
