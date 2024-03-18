# -*- coding: utf-8 -*-
# @Time    : 8/9/2021 12:10 PM
# @Author  : Paulo Radatz
# @Email   : pradatz@epri.com
# @File    : smart_inverter_functions.py
# @Software: PyCharm


class SmartInverterFunction:

    def __init__(self, dss, bus, kV_base, kW_rated, kVA, smart_inverter_function):

        self.dss = dss
        self.bus = bus
        self.kV_base = kV_base
        self.kW_rated = kW_rated
        self.kVA = kVA
        self.smart_inverter_function = smart_inverter_function

        self.__define_3ph_pvsystem_with_transformer()

        self.__define_smart_inverter_function()

    def __define_3ph_pvsystem_with_transformer(self):
        self.__define_transformer()

        self.dss.text("makebuslist")
        self.dss.text(f"setkVBase bus=PV_{self.bus} kVLL=0.48")

        self.__define_pvsystem_curves()

        self.__define_pvsystem()

    def __define_smart_inverter_function(self):
        if self.smart_inverter_function == "unity-pf":
            pass
        elif self.smart_inverter_function == "pf":
            self.__set_pf()
        elif self.smart_inverter_function == "volt-var":
            self.__set_vv()

    def __define_transformer(self):
        self.dss.text(
            f"New transformer.PV_{self.bus} "
            f"phases=3 "
            f"windings=2 "
            f"buses=({self.bus}, PV_{self.bus}) "
            f"conns=(wye, wye) "
            f"kVs=({self.kV_base}, 0.48) "
            f"xhl=5.67 "
            f"%R=0.4726 "
            f"kVAs=({self.kVA}, {self.kVA})")

    def __define_pvsystem_curves(self):
        self.dss.text("New XYCurve.MyPvsT "
                      "npts=4  "
                      "xarray=[0  25  75  100]  "
                      "yarray=[1.0 1.0 1.0 1.0]")

        self.dss.text("New XYCurve.MyEff "
                      "npts=4  "
                      "xarray=[.1  .2  .4  1.0]  "
                      "yarray=[1.0 1.0 1.0 1.0]")

        self.dss.text(
            "New Loadshape.MyIrrad "
            "npts=24 "
            "interval=1 "
            "mult=[0 0 0 0 0 0 .1 .2 .3  .5  .8  .9  1.0  1.0  .99  .9  .7  .4  .1 0  0  0  0  0]")

        self.dss.text(
            "New Tshape.MyTemp "
            "npts=24 "
            "interval=1 "
            "temp=[25, 25, 25, 25, 25, 25, 25, 25, 35, 40, 45, 50  60 60  55 40  35  30  25 25 25 25 25 25]")

    def __define_pvsystem(self):
        self.dss.text(
            f"New PVSystem.PV_{self.bus} "
            f"phases=3 "
            f"conn=wye  "
            f"bus1=PV_{self.bus} "
            f"kV=0.48 "
            f"kVA={self.kVA} "
            f"Pmpp={self.kW_rated} "
            f"pf=1 "
            f"%cutin=0.00005 "
            f"%cutout=0.00005 "
            f"VarFollowInverter=yes "
            f"effcurve=Myeff  "
            f"P-TCurve=MyPvsT "
            f"Daily=MyIrrad  "
            f"TDaily=MyTemp "
            f"wattpriority=False")

    def __set_pf(self):
        self.dss.text(f"edit PVSystem.PV_{self.bus} pf=-0.90 pfpriority=True")

    def __set_vv(self):
        x_vv_curve = "[0.5 0.92 0.98 1.0 1.02 1.08 1.5]"
        y_vv_curve = "[1 1 0 0 0 -1 -1]"
        self.dss.text(f"new XYcurve.volt-var npts=7 yarray={y_vv_curve} xarray={x_vv_curve}")
        self.dss.text(
            "new invcontrol.inv "
            "mode=voltvar "
            "voltage_curvex_ref=rated "
            "vvc_curve1=volt-var "
            "RefReactivePower=VARMAX")


