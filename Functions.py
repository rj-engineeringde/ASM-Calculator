import sys 
from typing import Any
from tabulate import tabulate
import math 
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, diff
from sympy import lambdify
import copy
import streamlit as st
import pandas as pd
import re

#####################################################################
# Define the motor class
#####################################################################

class MotorAsm:
    def __init__(self, 
                 Pn: float,             # Nominal Power [kW]
                 Un: int,               # Nominal Voltage [V]
                 Freq: int,             # Nominal Frequency [Hz]
                 n: int,                # Rotational speed [RPM]
                 eta: float,            # Efficiency [%]
                 cosphi: float,         # Cosinus Phi
                 Ia: float,             # Starting Current [%]
                 Ma: float,             # Starting Torque [%]
                 Mk: float,             # Maximum Torque [%]
                 connection: str,       # Connection (Y, D)
                 deltaT: float,         # Temperature Rise [K]
                 ambientTemp: int,      # Ambient Temperature [°C]
                 ambientMeter: int,     # Operation Height Above Sea Level [m]
                 no_parallel: int,      # Number of Parallel Circuits on the Stator
                 rotorVoltage: float,   # Rotor Voltage [V]
                 motor_label: str       # Label of the motor for plots
                 ):
        self.Pn             = Pn                
        self.Un             = Un                
        self.Freq           = Freq              
        self.n              = n                 
        self.eta            = eta               
        self.cosphi         = cosphi            
        self.Ia             = Ia                
        self.Ma             = Ma                
        self.Mk             = Mk                
        self.connection     = connection        
        self.deltaT_l       = deltaT  # Temperature Rise [K], assumed to increase lineary with the current in the coils
        self.deltaT_q       = deltaT  # Temperature Rise [K], assumed to increase quadratic with the current in the coils
        self.ambientTemp    = ambientTemp       
        self.ambientMeter   = ambientMeter      
        self.no_parallel    = no_parallel       
        self.rotorVoltage   = rotorVoltage  
        if rotorVoltage > 0:
            self.update_rotor_current()
        else:
            self.rotorCurrent = 0
        self.rotorConnection = ''
        self.motor_label    = motor_label 

        self.In = self.get_In(self.Pn, self.cosphi, self.eta, self.Un)
        self.Mn = self.get_Mn(self.Pn, self.n)
        self.Ia_abs = self.Ia / 100 * self.In
        self.Ma_abs = self.Ma / 100 * self.Mn
        self.Mk_abs = self.Mk / 100 * self.Mn
    
    # Nennleistung verändern (umstempeln) ohne andere Parameter zu verändern
    def variate_power(self, Pn_new: float) -> str:
        """
        *** Return [Str]: Explenation of the conducted calculations *** \n
        Operate the motor with a different power.\n
        Frequency=Const., Voltage=Const.
        """
        # Cache old values
        Pn_old = self.Pn
        In_old = self.In
        Ia_old = self.Ia
        Ma_old = self.Ma
        Mk_old = self.Mk

        if round(Pn_new) == round(self.Pn):
            return ''
        else:
            # Calculate the new nominal torque and current from the new Power
            self.Mn = self.get_Mn(Pn_new, self.n)
            self.In = self.get_In(Pn_new, self.cosphi, self.eta, self.Un)

            # Update the %-values of Ia, Ma, Mn
            self.Ia = self.Ia_abs / self.In * 100
            self.Ma = self.Ma_abs / self.Mn * 100
            self.Mk = self.Mk_abs / self.Mn * 100

            # Update the temperature Rise
            dT_returnstring = self.update_deltaT(In_old, self.In)

            # Update the Power value
            self.Pn = Pn_new

            # Return explenation of conducted calculation
            returnstring =  f"\n\nDifferent Power {round(Pn_old)}: kW --> {round(self.Pn)} kW \n" \
                            f"   In_new = {round(self.Pn)} kW * 1000 / ( sqrt{{3}} * {round(self.cosphi, 2)} * {round(self.eta / 100, 4)} * {round(self.Un)} V ) = {round(self.In, 1)} A \n" \
                            f"   Mn_new = {round(self.Pn)} kW * 1000 / ( 2 * pi * {round(self.n)} RPM / 60 ) = {round(self.Mn)} Nm \n" \
                            f"   Ia_new = {round(Pn_old)} kW / {round(self.Pn)} kW * {round(Ia_old)} % = {round(self.Ia)} % \n" \
                            f"   Ma_new = {round(Pn_old)} kW / {round(self.Pn)} kW * {round(Ma_old)} % = {round(self.Ma)} % \n" \
                            f"   Mk_new = {round(Pn_old)} kW / {round(self.Pn)} kW * {round(Mk_old)} % = {round(self.Mk)} % \n" \
                            f"{dT_returnstring}"
            return returnstring
        
    # Frequenz und Spannung im gleichen Maße erhöhen/veringern. Magnetischer Fluss bleibt konstant
    def variate_freq_volt_konstMagnFlux(self, Freq_new: int) -> str:
        """
        *** Return [Str]: Explenation of the conducted calculations *** \n
        Increase/decrease frequency and voltage by the same factor. \n
        Magnetic_Flux = Const., U/Freq=Const. \n
        Nominal power and RPM increase by the factor Freq_new/Freq_old \n
        Nominal, max. and starting torque stay unchanged \n
        Nominal and starting current stay unchanged
        """
        # Cache old values
        Freq_old = self.Freq
        Pn_old = self.Pn
        Un_old = self.Un
        n_old = self.n

        if round(Freq_new) == round(self.Freq):
            return ''
        else:
            # Update the values of Pn, Un, n, Freq
            factor = Freq_new / self.Freq
            self.Pn = factor * self.Pn
            self.n = factor * self.n
            self.Un = factor * self.Un
            self.Freq = Freq_new

            # Return explenation of conducted calculation
            returnstring =  f"\n\nDifferent Frequency & Voltage with const. U/f: \n{round(Freq_old)} Hz, {round(Un_old)} V --> {round(self.Freq)} Hz, {round(self.Un)} V: \n" \
                            f"   factor = {round(self.Freq)} Hz / {round(Freq_old)} Hz = {round(factor, 2)} \n" \
                            f"   Pn_new = {round(factor, 2)} * {round(Pn_old)} kW = {round(self.Pn)} kW\n" \
                            f"   n_new = {round(factor, 2)} * {round(n_old)} RPM = {round(self.n)} RPM"
            return returnstring

    # Spannung erhöhen / Veringern
    def variate_voltage(self, Un_new: int) -> str:
        """
        *** Return [Str]: Explenation of the conducted calculations *** \n
        Increase/decrease voltage \n
        Nominal power and frequency stay unchanged \n
        Starting values (torque, current), max. torque and nominal current are afected \n
        Temperature Rise is afected
        """
        # Cache old values
        Un_old = self.Un
        In_old = self.In
        Ia_old = self.Ia
        Ma_old = self.Ma
        Mk_old = self.Mk
        Ia_abs_old = self.Ia_abs

        if round(Un_new) == round(self.Un):
            return ''
        else:
            # Calculate new Ia, Ma, Mk
            factor = Un_new / Un_old
            self.Un = Un_new
            self.Ma = factor**2 * Ma_old # Starting torque [%] proportional to U^2
            self.Mk = factor**2 * Mk_old # Maximum Torque [%] proportional to U^2
            self.In = self.get_In(self.Pn, self.cosphi, self.eta, self.Un) # Update the value of In
            self.Ia_abs = factor * Ia_abs_old # Starting current [A] proportional to U
            self.Ia = self.Ia_abs / self.In * 100
            self.refresh_abs_Ia_Ma_Mn() # Ubdate the absolute values of Ia_abs, Ma_abs, Mn_abs

            # Calculate new deltaT
            dT_returnstring = self.update_deltaT(In_old, self.In)

            # Return explenation of conducted calculation
            returnstring =  f"\n\nDifferent Voltage: {round(Un_old)} V --> {round(self.Un)} V \n" \
                            f"   In_new = {round(self.Pn)} kW * 1000 / ( sqrt{{3}} * {round(self.cosphi, 2)} * {round(self.eta / 100, 4)} * {round(self.Un)} V ) = {round(self.In, 1)} A \n" \
                            f"   Ia_abs_new = {round(Un_new)} V / {round(Un_old)} V * {round(Ia_abs_old)} A = {round(self.Ia_abs)} A \n" \
                            f"   Ia_new = {round(self.Ia_abs)} A / {round(self.In, 2)} A = {round(self.Ia)} % \n" \
                            f"   Ma_new = ( {round(Un_new)} V / {round(Un_old)} V )^2 * {round(Ma_old)} % = {round(self.Ma)} % \n" \
                            f"   Mk_new = ( {round(Un_new)} V / {round(Un_old)} V )^2 * {round(Mk_old)} % = {round(self.Mk)} % \n" \
                            f"{dT_returnstring}"
                            #f"Ia_new = ( {round(Un_new)} V / {round(Un_old)} V * {round(Ia_abs_old)} A ) / ({round(self.In, 2)} A) = {round(self.Ia)} % \n" \
            return returnstring

    # Motor in D / Y umschalten
    def variate_connection(self, connection_new: str) -> str:
        """
        *** Return [Str]: Explenation of the conducted calculations *** \n
        Connect stator as Y or D \n
        Nominal power, frequency, starting %-values and max. torque stay unchanged \n
        Nominal current and voltage are afected
        """
        In_old = self.In
        Un_old = self.Un
        try:
            if connection_new == self.connection:
                return ''
            elif connection_new == "Y":
                self.In = self.In / math.sqrt(3) # Current increases by sqrt(3) from D to Y
                self.Un = self.Un * math.sqrt(3) # Voltage decreases by sqrt(3) from D to Y
                self.refresh_abs_Ia_Ma_Mn() # Refresh absolute starting current, since the nominal current changed
                self.connection = connection_new # Update connection string value
                txt_return = f"""\n\nConnection D --> Y:\n   In_new = {round(In_old, 1)} A / sqrt{{3}} = {round(self.In, 1)} A\n   Un_new = {round(Un_old)} V * sqrt{{3}} = {round(self.Un)} V"""
                return txt_return
            elif connection_new == "D":
                self.In = self.In * math.sqrt(3) # Current decreases by sqrt(3) from Y to D
                self.Un = self.Un / math.sqrt(3) # Voltage increases by sqrt(3) from Y to D
                self.refresh_abs_Ia_Ma_Mn() # Refresh absolute starting current, since the nominal current changed
                self.connection = connection_new # Update connection string value
                return f"""\n\nConnection Y --> D:\n   In_new = {round(In_old, 1)} A * sqrt{{3}}  = {round(self.In, 1)} A\n   Un_new = {round(Un_old)} V / sqrt{{3}} = {round(self.Un)} V"""
            else:
                raise # Raise Exception
        except Exception as e:
            print("\nERROR: Class: MotorAsm; Function: variate_connection() --> Wrong Input:", connection_new)
            sys.exit(1)

    # Motor in Parallel schalten oder andere Kombination wählen
    def variate_number_of_parallel_circuits(self, no_parallel_new: int) -> str:
        """
        *** Return [Str]: Explenation of the conducted calculations *** \n
        Change the connection of the branches of the stator. E.g. two branches in parallel \n
        Nominal power, frequency, starting %-values and max. torque stay unchanged \n
        Nominal current and voltage are afected
        """
        # Define old values
        Un_old = self.Un 
        In_old = self.In
        no_parallel_old = self.no_parallel

        if no_parallel_new == self.no_parallel:
            return ''
        else:
            # Calculate new voltage and current
            self.no_parallel = no_parallel_new
            self.Un = Un_old * (no_parallel_old / no_parallel_new)
            self.In = In_old * (no_parallel_new / no_parallel_old)
            self.refresh_abs_Ia_Ma_Mn()
            
            # Return explenation of conducted calculation
            returnstring =  f"\n\nChange Number of Parallel Branches on the Stator: {round(no_parallel_old)} --> {round(no_parallel_new)} : \n" \
                            f"   Un = {round(Un_old)} V * ( {round(no_parallel_old)} / {round(no_parallel_new)} ) = {round(self.Un)} V \n" \
                            f"   In = {round(In_old, 1)} A * ( {round(no_parallel_new)} / {round(no_parallel_old)} ) = {round(self.In, 1)} A" 
            return returnstring
    
    # Aufstellhöhe variieren
    def variate_ambient_height(self, ambient_height_new):
        """
        *** Return [Str]: Explenation of the conducted calculations *** \n
        Change the operating height above sea level. Correct DeltaT accordingly \n
        According to IEC60034-1:\n
        - If Machine tested at <=1000m and operation >1000m ==> dT_test = dT_operation * (1 - [H - 1000m] / 10000m)\n
        - If Machine tested at >1000m and operation <=1000m ==> dT_test = dT_operation * (1 + (H_Test - 1000m)/10000m)\n
        - If Machine tested at >1000m and operation >1000m ==> dT_test = dT_operation * (1 + (H_Test - H)/10000m)\n
        - If Machine tested at >4000m or operation >4000m ==> To be agreed. No reference from IEC60034-1
        """
        return_str = ''
        dT_lin_old = self.deltaT_l
        dT_quad_old = self.deltaT_q
        ambient_height_old = self.ambientMeter
        self.ambientMeter = ambient_height_new
        if ambient_height_old <= 1000:
            if ambient_height_new > 1000:
                self.deltaT_l = self.deltaT_l / (1 - (ambient_height_new - 1000)/10000)
                self.deltaT_q = self.deltaT_q / (1 - (ambient_height_new - 1000)/10000)
                return_str += f"\n\nDifferent heigt above sea level: {round(ambient_height_old)} m --> {round(ambient_height_new)} m\n"
                return_str += f"   dT_linear = {round(dT_lin_old)} K / ( 1 - ( {ambient_height_new} m - 1000 m ) / 10000 m ) = {round(self.deltaT_l, 1)} K\n"
                return_str += f"   dT_quadratic = {round(dT_quad_old)} K / ( 1 - ( {ambient_height_new} m - 1000 m ) / 10000 m ) = {round(self.deltaT_q, 1)} K\n"
            else:
                pass # If Operation at <1000m and Testing at <1000m, no correction needed
        elif ambient_height_old > 1000:
            if ambient_height_new < 1000:
                self.deltaT_l = self.deltaT_l / (1 + (ambient_height_old - 1000)/10000)
                self.deltaT_q = self.deltaT_q / (1 + (ambient_height_old - 1000)/10000)
                return_str += f"\n\nDifferent heigt above sea level: {round(ambient_height_old)} m --> {round(ambient_height_new)} m\n"
                return_str += f"   dT_linear = {round(dT_lin_old)} K / ( 1 + ( {ambient_height_old} m - 1000 m ) / 10000 m ) = {round(self.deltaT_l, 1)} K \n"
                return_str += f"   dT_quadratic = {round(dT_quad_old)} K / ( 1 + ( {ambient_height_old} m - 1000 m ) / 10000 m ) = {round(self.deltaT_q, 1)} K \n"
            else:
                self.deltaT_l = self.deltaT_l / (1 + (ambient_height_old - ambient_height_new)/10000)
                self.deltaT_q = self.deltaT_q / (1 + (ambient_height_old - ambient_height_new)/10000)
                return_str += f"\n\nDifferent heigt above sea level: {round(ambient_height_old)} m --> {round(ambient_height_new)} m\n"
                return_str += f"   dT_linear = {round(dT_lin_old)} / ( 1 + ( {ambient_height_old} m - {ambient_height_new} m ) / 10000 m ) = {round(self.deltaT_l, 1)} K\n"
                return_str += f"   dT_quadratic = {round(dT_quad_old)} / ( 1 + ( {ambient_height_old} m - {ambient_height_new} m ) / 10000 m ) = {round(self.deltaT_q, 1)} K\n"
        else:
            raise ValueError(f'Error: function variate_ambient_height() --> Wrong value {ambient_height_old=}')
        return return_str
    
    # Umgebungstemperatur variieren
    def variate_ambient_temp(self, ambient_temp_new):
        """
        *** Return [Str]: Explenation of the conducted calculations *** \n
        Change the operating temperature. Correct DeltaT accordingly \n
        According to IEC60034-1 Tab. 9, 1a and 1c: \n
        - deltaT_limit = deltaT_limit_old - (T_ambient_new - T_ambient_old) \n
        - Exception: For 0<= Temp_ambient <= 40 and height <1000m: If Temp. Class - (Limit Temp. Rise + 40°C) > 5K, Correction factor is needed\n
                --> I ignored this exception \n
                --> Correction Factor: (1 - (thermal class - (40°C + limit temp. Rise)) / 80K) \n
        Alternative Method in catalogue: Multiply by factor (5% per 5°C). This gives similar results, for deltaT<100K a bit lower correction.\n
        I choosed the option between both, that gives the higher deltaT, since this is a more conservative value.
        """
        return_str = ''
        dT_lin_old = self.deltaT_l
        dT_quad_old = self.deltaT_q
        ambient_temp_old = self.ambientTemp
        self.ambientTemp = ambient_temp_new
        
        # Correction by substraction of temp. from deltaT
        add_deltaT_l = self.deltaT_l + (ambient_temp_new - ambient_temp_old)
        add_deltaT_q = self.deltaT_q + (ambient_temp_new - ambient_temp_old)

        # Correction by factor
        factor = 1 + (ambient_temp_new - ambient_temp_old)/100
        fac_deltaT_l = factor * self.deltaT_l
        fac_deltaT_q = factor * self.deltaT_q

        # Result
        self.deltaT_l = max([add_deltaT_l, fac_deltaT_l])
        self.deltaT_q = max([add_deltaT_q, fac_deltaT_q])

        # Return String
        if round(ambient_temp_new) != round(ambient_temp_old):
            return_str += f"\n\nDifferent ambient temperature: {round(ambient_temp_old)} °C --> {round(ambient_temp_new)} °C\n"
            if add_deltaT_l > fac_deltaT_l:
                return_str += f"   dT_linear = {round(dT_lin_old, 1)} K + ( {ambient_temp_new} °C - {ambient_temp_old} ) = {round(self.deltaT_l, 1)} K\n"
            else:
                return_str += f"   dT_linear = {round(dT_lin_old, 1)} K * {factor} = {round(self.deltaT_l, 1)} K\n"
            if add_deltaT_q > fac_deltaT_q:
                return_str += f"   dT_quadratic = {round(dT_quad_old, 1)} K + ( {ambient_temp_new} °C - {ambient_temp_old} ) = {round(self.deltaT_q, 1)} K"
            else:
                return_str += f"   dT_quadratic = {round(dT_quad_old, 1)} K * {factor} = {round(self.deltaT_q, 1)} K"
        return return_str

    # Rotor in D/Y umschalten
    def variate_connection_rotor(self, rotorChangeConnection: str) -> str:
        '''Variate the connection of the rotor \n
        - Input: rotorChangeConnection <Str> - Expected: "Do not change", "Y --> D", "D --> Y"
        - Return: String with calcultation
        '''
        str_calc = ''
        rotorVoltage_old = self.rotorVoltage
        # Change: Y --> D
        if rotorChangeConnection == 'Y --> D':
            self.rotorVoltage = round(self.rotorVoltage / math.sqrt(3))
            self.rotorConnection = 'D'
            str_calc += f"\n\nChange Rotor Connection: Y --> D\n"
            str_calc += f"   Un Rotor = {round(rotorVoltage_old)} V / sqrt{{3}}  = {self.rotorVoltage} V"

        # Change: D --> Y
        elif rotorChangeConnection == 'D --> Y':
            self.rotorVoltage = round(self.rotorVoltage * math.sqrt(3))
            self.rotorConnection = 'Y'
            str_calc += f"\n\nChange Rotor Connection: D --> Y\n"
            str_calc += f"   Un Rotor = {round(rotorVoltage_old)} V * sqrt{{3}} = {self.rotorVoltage} V"
        
        # If any other input, do not variate connection
        else:
            self.rotorConnection = ''
            pass

        # Return New Voltage and calculation
        return str_calc

    # Variate the rotor Voltage
    def variate_voltage_rotor(self, statorVoltage_ini: float, statorVoltage_res: float) -> str:
        '''
        Variate rotor voltage from a change in stator voltage (Un_Rotor ~ Un_Stator)\n
        Update self.rotorVoltage accordingly\n
        Return String with explenation
        '''
        rotorVoltage_old = self.rotorVoltage
        str_calc = ''
        if round(statorVoltage_ini) != round(statorVoltage_res):
            self.rotorVoltage = round(rotorVoltage_old * statorVoltage_res / statorVoltage_ini)
            str_calc += f"\n\nUpdate Rotor Voltage: U_Rotor ~ U_Stator\n"
            str_calc += f"   Un Rotor = {round(rotorVoltage_old)} V * {round(statorVoltage_res)} V / {round(statorVoltage_ini)} V = {self.rotorVoltage} V"
        return str_calc

    # Update the rotor current
    def update_rotor_current(self) -> str:
        '''
        Update the self.rotorCurrent from self.Pn, self.rotorVoltage\n
        Return [String]: Calculation explenation
        '''
        self.rotorCurrent = round(self.Pn * 1000 * 1.1 / (self.rotorVoltage * math.sqrt(3)), 1)
        str_calc = f"\n\nCalculate Rotor Current:\n"
        str_calc += f"   In Rotor = ( {round(self.Pn, 1)} kW * 1000 * 1.1 ) / ( {round(self.rotorVoltage)} V * sqrt{{3}} ) = {self.rotorCurrent} A"
        return str_calc

    # In berechnen
    def get_In(self, Pn: float, cosphi: float, eta: float, Un: int) -> float:
        """
        *** Return [Float]: Calculated nominal current *** \n
        Calculate the nominal current of the motor from the input parameters (nominal power, cos_phi, efficiency, nominal voltage)
        """
        In = Pn * 1000 / ( math.sqrt(3) * cosphi * eta / 100 * Un )
        return In

    # Mn berechnen
    def get_Mn(self, Pn: float, n: int) -> float:
        """
        *** Return [Float]: Calculated nominal torque *** \n
        Calculate the nominal torque of the motor from the input parameters (nominal power, nominal speed)
        """
        Mn = Pn * 1000 / ( 2*3.1415926536*n/60 ) 
        return Mn

    # Aktualisiere die Absolutwerte anhand der Prozentangaben von Ia, Ma, Mk
    def refresh_abs_Ia_Ma_Mn(self) -> None:
        """
        *** Return [None]: None *** \n
        Update the absolute starting current, starting torque and max. torque from the percentual values and nominal values
        """
        self.Ia_abs = self.Ia / 100 * self.In
        self.Ma_abs = self.Ma / 100 * self.Mn
        self.Mk_abs = self.Mk / 100 * self.Mn

    # Aktualisiert die Werte von deltaT in Abhängigkeit von einer Stromänderung
    def update_deltaT(self, In_old: float, In_new: float) -> str:
        """
        *** Return [Str]: Explenation of the conducted calculations *** \n
        Update the value of temperature rise by an increase/decrease in current \n
        Assumed relations: dT prop. I, dT prop. I^2
        """
        deltaT_l_old = self.deltaT_l
        deltaT_q_old = self.deltaT_q

        self.deltaT_l = In_new / In_old * self.deltaT_l # Assumption: deltaT ~ I 
        self.deltaT_q = (In_new / In_old)**2 * self.deltaT_q # Assumption: deltaT ~ I^2
        
        returnstring = f"   dT_linear = {round(In_new, 1)} A / {round(In_old, 1)} A * {round(deltaT_l_old)} K = {round(self.deltaT_l)} K \n" \
                       f"   dT_quadratic = ( {round(In_new, 1)} A / {round(In_old, 1)} A )^2 * {round(deltaT_q_old)} K = {round(self.deltaT_q)} K"
        return returnstring
    
    # Strom in einem Strang des Stators berechnen
    def get_branch_voltage_current(self) -> float | int:
        """
        *** Return [Float, Int]: I_branch, U_branch *** \n
        Calculate the current and voltage in a single branch of the stator
        """
        # Calculate the voltage and current in a single phase
        if self.connection == "Y":
            U_branch = self.Un / math.sqrt(3)
            I_phase = self.In
        elif self.connection == "D":
            U_branch = self.Un
            I_phase = self.In / math.sqrt(3)
        else:
            raise ValueError("ERROR: Function: get_branch_voltage_current() --> Connection must be 'Y' or 'D'")

        # Calculate the current and voltage in a single branch of a single phase
        I_branch = I_phase / self.no_parallel
        return I_branch, U_branch
    
    # Print Values of the motor in a table
    def print_motor_values(self, show_print: bool=True) -> str:
        """
        *** Return [Str]: Table String *** \n
        Print the motor parameters in a table
        """
        print_text = tabulate(
                [
                    ["Nominal Power [kW]", round(self.Pn)],
                    ["Nominal Voltage [V]", round(self.Un)],
                    ["Nominal Frequency [Hz]", round(self.Freq)],
                    ["Nominal Speed [min-1]", round(self.n)],
                    ["Efficiency [%]", round(self.eta, 2)],
                    ["Cosinus Phi", round(self.cosphi, 2)],
                    ["Temperature Rise (dT~I) [K]", round(self.deltaT_l)],
                    ["Temperature Rise (dT~I^2) [K]", round(self.deltaT_q)],
                    ["", ""],
                    ["Connection (Y, D)", self.connection],
                    ["Number of Parallel Circuits (Stator)", self.no_parallel],
                    ["", ""],
                    ["Nominal Current [A]", round(self.In, 1)],
                    ["Starting Current [%]", round(self.Ia)],
                    ["Starting Current [A]", round(self.Ia_abs)],
                    ["", ""],
                    ["Nominal Torque [Nm]", round(self.Mn)],
                    ["Starting Torque [%]", round(self.Ma)],
                    ["Starting Torque [Nm]", round(self.Ma_abs)],
                    ["Maximum Torque [%]", round(self.Mk)],
                    ["Maximum Torque [Nm]", round(self.Mk_abs)],
                    ["", ""],
                    ["Current in a Single Stator Branch [A]", round(self.get_branch_voltage_current()[0], 1)],
                    ["Voltage in a Single Stator Branch [V]", round(self.get_branch_voltage_current()[1])],
                    ["", ""],
                    ["Ambient Temperature [°C]", self.ambientTemp],
                    ["Operation Height [m]", self.ambientMeter],
                    ["", ""],
                    ["Rotor Voltage [V]", round(self.rotorVoltage)],
                ],
                headers=['Variable', 'Value'], 
                tablefmt='outline'
            )
        
        if show_print:
            print(print_text)

        return print_text
    
    # Get M_n curve of the motor
    def get_M_n_curve(self) -> Any:
        """
        *** Return [Lambda function]: M(n) curve *** \n
        Calculate an approximated M-n-curve of the motor
        """
        # Get synchrone speed
        n_sync = get_n_synchrone(self.n, self.Freq)

        x = symbols('x')
        slip_at_Mk = symbols('slip_at_Mk')

        # Define function | Kloss-like equation in terms of speed
        M = 2 * self.Mk_abs / ( (n_sync * slip_at_Mk) / (n_sync - x) + (n_sync - x) / (n_sync * slip_at_Mk) )

        # Equation to solve
        eq1 = Eq(M.subs(x, 0.1), self.Ma_abs)

        # Solve equation
        solution = solve([eq1], (slip_at_Mk))
        slip_at_Mk_solution = min(solution)[0] # Choose the lower value of slip_at_Mk. The equation gets two results, one of them >1. A slip >1 does not make sense foran application as motor.

        # Lambdify the symbolic expression for fast numeric evaluation
        return lambdify(x, M.subs(slip_at_Mk, slip_at_Mk_solution), modules='numpy')
    
    # Get I_n curve of the motor
    def get_I_n_curve(self, Ia_type: str = 'total') -> Any:
        """
        *** Return [Lambda function]: I(n) curve *** \n
        *** Input [str]: 'total' / 'branch' *** \n
            --> Consider the total current of the motor ('total') or the one in a single branch ('branch')
        Calculate an approximated I-n-curve of the motor 
        """
        # Get synchrone speed
        n_sync = get_n_synchrone(self.n, self.Freq)

        x = symbols('x')
        k = symbols('k')

        if Ia_type == 'total':
            Ia_abs = self.Ia_abs
        elif Ia_type == 'branch':
            Ia_abs = self.get_branch_voltage_current()[0] * self.Ia / 100
        else:
            raise ValueError("ERROR: Function: get_I_n_curve() --> Current Type must be 'total' or 'branch'")

        # Define function and its derivates
        I = Ia_abs * ( n_sync / (n_sync - x) )**k

        # Equation to solve
        eq1 = Eq(I.subs(x, self.n), self.In)

        # Solve equation
        k_solution = solve([eq1], k)[0][0]

        # Lambdify the symbolic expression for fast numeric evaluation
        return lambdify(x, I.subs(k, k_solution), modules='numpy')


#####################################################################
# Define the functions that are not part of the motor class
#####################################################################

# Exctract the numerical part of a string. Return <None> if no numerical data was extracted 
def extract_numeric(value: str) -> float:
    """Convert String value to numeric float\n
    Return: \n
    - Float if convesion possible\n
    - None if conversion not possible"""
    if pd.isna(value):
        return None
    # Replace commas with dots
    value = str(value).replace(',', '.')
    # Extract the first number-like pattern (int or float)
    match = re.search(r"[-+]?\d*\.?\d+", value)
    return float(match.group()) if match else None

# Get synchronus speed from nominal speed
def get_n_synchrone(n: int, freq: int) -> int:
    """
    *** Return [Int]: Synchrone Speed [RPM] *** \n
    Calculate the synchrone speed for a given nominal speed
    """
    slip_range = (0, 0.1)  # typical slip range
    for poles in range(2, 16, 2):  # iterate over the possible poles
        n_sync = 120 * freq / poles # Calculate the synchronus speed for that pole number
        n_min = n_sync * (1 - slip_range[1])
        n_max = n_sync * (1 - slip_range[0])
        if n_min <= n <= n_max:
            return n_sync
    raise ValueError("ERROR: Function: get_n_synchron() --> no pole number was detected. n_nominal={}".format(n))

# Plot starting curves
def plot_asm_start_curves(motors: list[MotorAsm], plt_show: bool=True) -> plt.figure:
    '''
    Generate a plot of a list of motors containing the starting curves (M_n, I_n, I_n for 1 Branch)\n
    Inputs:\n
    - motors: List of [MotorAsm] class motors\n
    - plt_show: Show plot plt.show()\n
    Outputs:\n
    - figure: plt,figure
    '''
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # Right y-axis for current

    line_colors = ['black', 'brown', 'darkgreen', 'peru', 'orangered', 'darkmagenta']
    line_styles = ['-', '-', '--', '--', '--', '--']
    dot_colors = ['red', 'green', 'blue', 'orange', 'purple', 'darkviolet']

    # Get nominal current and power for reference of the axis
    Mn_axis_define = motors[0].Mn
    In_axis_define = motors[0].In
    In_axis_define_branch = motors[0].get_branch_voltage_current()[0]
    
    # Legend Variables
    lines = []
    dots = []
    
    for idx, motor in enumerate(motors):
        # Get synchrone speed
        n_sync = get_n_synchrone(motor.n, motor.Freq)

        # Get the M-n-curve and I-n-curve for the motor as lambda function
        M_func = motor.get_M_n_curve()
        I_func = motor.get_I_n_curve()
        I_func_branch = motor.get_I_n_curve(Ia_type='branch')

        # Generate x (speed) values and evaluate M(x), I(x)
        x_vals = np.linspace(0.1, n_sync * 0.9999, 100)  # Avoid division by zero at x=n_sync
        M_vals = M_func(x_vals) / Mn_axis_define * 100 # Values are plotted in %
        I_vals = I_func(x_vals) / In_axis_define * 100 # Values are plotted in %
        I_vals_branch = I_func_branch(x_vals) / In_axis_define_branch * 100 # Values are plotted in %

        # Distinguish motors with different linestyles or colors
        motor_label = motor.motor_label

        # Plot torque (left y-axis)
        line1, = ax1.plot(x_vals, M_vals, label=f"{motor_label}", color=line_colors[idx], linestyle=line_styles[idx])
        dot1, = ax1.plot(motor.n, motor.Mn/Mn_axis_define*100, 'o', label=f"{motor_label} | Mn", color=dot_colors[idx])

        # Plot Current (right y-axis)
        ax2.plot(x_vals, I_vals, color=line_colors[idx], linestyle=line_styles[idx]) #label=f"{motor_label} - Current"
        dot2, = ax2.plot(motor.n, motor.In/In_axis_define*100, '^', label=f"{motor_label} | In", color=dot_colors[idx])

        if motor.In != motor.get_branch_voltage_current()[0]:
            # Plot current in a single Branch
            line2, = ax2.plot(x_vals, I_vals_branch, label=f"{motor_label} (per Stator Branch)", color=line_colors[idx], linestyle='--')
            dot3, = ax2.plot(motor.n, motor.get_branch_voltage_current()[0]/In_axis_define_branch*100, '+', label=f"{motor_label} | In_branch", color=dot_colors[idx])

        # Combine Legends
        lines.append(line1)
        dots.append(dot1)
        dots.append(dot2)
        if motor.In != motor.get_branch_voltage_current()[0]:
            dots.append(dot3)
            lines.append(line2)
    
    # Axis labels and grid
    ax1.set_xlabel('Speed [RPM]')
    ax1.set_ylabel('Torque [%]', color='black')
    ax2.set_ylabel('Current [%]', color='midnightblue')
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='midnightblue')
    ax1.grid(True, linestyle=':', color='black')
    ax2.grid(True, linestyle='--', color='midnightblue')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    # Combine legends
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    labels_lines = [line.get_label() for line in lines]
    labels_dots = [dot.get_label() for dot in dots]
    ax2.legend(lines + dots, 
               labels_lines + labels_dots, 
               loc='lower center', 
               framealpha=1, 
               facecolor='white',
               bbox_to_anchor=(0.5, -0.4),  # Place legend below x-axis
               ncol=3,  # Adjust columns as needed
               )
    plt.subplots_adjust(bottom=0.4)
    plt.title("Starting Curves")
    plt.tight_layout()

    if plt_show:
        plt.show()

    return fig

# Generate header text
def print_header(txt: str) -> str:
    sep_txt = '--------------------------------------------------------------------------'
    return f"\n{sep_txt} \n{txt} \n{sep_txt} \n"

# Calculate percentual changes between initial motor and result motor
def calculate_percentual_changes(motor_ini: MotorAsm, motor_res: MotorAsm) -> pd.DataFrame:
    '''
    Inputs: \n
    - df_ini: Initial Dataframe | <class MotorAsm>\n
    - df_res: Result Dataframe  | <class MotorAsm>\n
    Output: \n
    - change_df: Percentual Changes | <pd.DataFrame>
    Calculate percentual changes of: P, B~U/f, Un_branch, In_branch, deltaT_lin, deltaT_quad
    '''
    In_ini_branch, Un_ini_branch = motor_ini.get_branch_voltage_current()
    In_res_branch, Un_res_branch = motor_res.get_branch_voltage_current()
    U_f_change = ( (Un_res_branch / motor_res.Freq) / (Un_ini_branch / motor_ini.Freq) - 1 )*100
    Un_change = ( Un_res_branch / Un_ini_branch - 1 )*100
    In_change = ( In_res_branch / In_ini_branch - 1 )*100
    Pn_change = ( motor_res.Pn / motor_ini.Pn - 1 )*100
    dT_lin_change = ( motor_res.deltaT_l / motor_ini.deltaT_l - 1 )*100
    dT_quad_change = ( motor_res.deltaT_q / motor_ini.deltaT_q  - 1 )*100

    change_df = pd.DataFrame({
        "Variable": ["Pn", "B (~U/f)", "Un (per branch)", "In (per branch)", "Temp. Rise (~I^2)", "Temp. Rise (~I)"],
        "Change": [round(Pn_change, 1), round(U_f_change, 1), round(Un_change, 1), round(In_change, 1), round(dT_quad_change, 1), round(dT_lin_change, 1)],
    })
    
    return change_df

# Make calculation of operating (result) values
def calculate_operating_values(
        # Machine initial values
        Pn: float,
        Un: int, 
        Freq: int, 
        ambientTemp: int, 
        ambientMeter: int, 
        connection: str,
        no_parallel: int, 
        Ia: int, 
        Ma: int, 
        Mk: int, 
        eta: float,
        cosphi: float,
        n: int,
        deltaT: float,
        rotorVoltage: float,
        rotorChangeConnection: str, # "Do not change", "Y --> D", "D --> Y"
        motor_label_ini: str,

        # Operating conditions
        Pn_op: float, 
        Un_op: int, 
        Freq_op: int, 
        ambientTemp_op: int, 
        ambientMeter_op: int, 
        connection_op: str,
        no_parallel_op: int, 
        motor_label_op: str
    ) -> tuple[pd.DataFrame, str, plt.Figure]:
    """
    Input: initial_values: <pd.DataFrame> \n
    .... "Pn [kW]" --> Float \n
    .... "Un [V]" --> Int \n
    .... "Freq [Hz]" --> Int \n
    .... "Ambient Temp. [°C]" --> Int \n
    .... "Height (m.a.s.l.) [m]" --> Int \n
    .... "Connection Y/D" --> Str ('Y'/'D') \n 
    .... "Parallel Branches (Stator)" --> Int \n
    .... "Temp. Rise [K]" --> Float \n
    .... "Ia/In [%]" --> Int \n
    .... "Ma/Mn [%]" --> Int \n
    .... "Mk/Mn [%]" --> Int \n
    .... "η [%]" --> Float \n
    .... "cos(φ)" --> Float \n
    .... "Nominal Speed [RPM]" --> Int \n
    ---> Initial parameters of the motor \n

    Input: operating_values: <pd.DataFrame> \n
    .... "Pn [kW]" --> Float \n
    .... "Un [V]" --> Int \n
    .... "Freq [Hz]" --> Int \n
    .... "Ambient Temp. [°C]" --> Int \n
    .... "Height (m.a.s.l.) [m]" --> Int \n
    .... "Connection Y/D" --> Str ('Y'/'D') \n 
    .... "Parallel Branches (Stator)" --> Int \n
    ---> Operating parameters of the motor \n

    Output: result: <pd.DataFrame> \n
    .... "Pn [kW]" --> Float \n
    .... "Un [V]" --> Int \n
    .... "Freq [Hz]" --> Int \n
    .... "Ambient Temp. [°C]" --> Int \n
    .... "Height (m.a.s.l.) [m]" --> Int \n
    .... "Connection Y/D" --> Str ('Y'/'D') \n 
    .... "Parallel Branches (Stator)" --> Int \n
    .... "Temp. Rise [K]" --> Float \n
    .... "Ia/In [%]" --> Int \n
    .... "Ma/Mn [%]" --> Int \n
    .... "Mk/Mn [%]" --> Int \n
    .... "η [%]" --> Float \n
    .... "cos(φ)" --> Float \n
    .... "Nominal Speed [RPM]" --> Int \n\n
    ---> Result of the calculations \n

    Output: calc_str: <String> \n
    ---> String containing the calculations conducted in a text format \n

    Output: plt_fig: <plt.Figure> \n
    ---> Figure of the plot for visualizing the calculations \n

    Output: change_df: <pd.DataFrame>\n
    ---> DataFrame containing the percentual changes of relevant values (e.g. U/f, P, I_branch, U_Branch, ...)
    """
    # Define relevant variables
    motor_ini = MotorAsm( 
                    Pn=Pn,                     # Nominal Power [kW]
                    Un=Un,                     # Nominal Voltage [V]
                    Freq=Freq,                 # Nominal Frequency [Hz]
                    n=n,                       # Rotational speed [RPM]
                    eta=eta,                   # Efficiency [%]
                    cosphi=cosphi,             # Cosinus Phi
                    Ia=Ia,                     # Starting Current [%]
                    Ma=Ma,                     # Starting Torque [%]
                    Mk=Mk,                     # Maximum Torque [%]
                    connection=connection,     # Connection (Y, D)
                    deltaT=deltaT,             # Temperature Rise [K]
                    ambientTemp=ambientTemp,   # Ambient Temperature [°C]
                    ambientMeter=ambientMeter, # Operation Height Above Sea Level [m]
                    no_parallel=no_parallel,   # Number of Parallel Circuits on the Stator
                    rotorVoltage=rotorVoltage, # Rotor Voltage [V]
                    motor_label=motor_label_ini # Label of the motor for plots
            )
    motor = MotorAsm( 
                    Pn=Pn,                     # Nominal Power [kW]
                    Un=Un,                     # Nominal Voltage [V]
                    Freq=Freq,                 # Nominal Frequency [Hz]
                    n=n,                       # Rotational speed [RPM]
                    eta=eta,                   # Efficiency [%]
                    cosphi=cosphi,             # Cosinus Phi
                    Ia=Ia,                     # Starting Current [%]
                    Ma=Ma,                     # Starting Torque [%]
                    Mk=Mk,                     # Maximum Torque [%]
                    connection=connection,     # Connection (Y, D)
                    deltaT=deltaT,             # Temperature Rise [K]
                    ambientTemp=ambientTemp,   # Ambient Temperature [°C]
                    ambientMeter=ambientMeter, # Operation Height Above Sea Level [m]
                    no_parallel=no_parallel,   # Number of Parallel Circuits on the Stator
                    rotorVoltage=rotorVoltage, # Rotor Voltage [V]
                    motor_label=motor_label_op # Label of the motor for plots
            )
    calculation_str = print_header("Initial Values") + motor_ini.print_motor_values(show_print=False)
    calculation_str_print = ''

    calculation_str += '\n\n' + print_header("Calculations")

    # Change Connection to Y / D
    txt_calc = motor.variate_connection(connection_op)
    calculation_str += txt_calc
    calculation_str_print += txt_calc
    
    # Change Nr. of Parallel Branches of the stator coils
    txt_calc = motor.variate_number_of_parallel_circuits(no_parallel_op)
    calculation_str += txt_calc
    calculation_str_print += txt_calc

    # Variate Ambient Temp.
    txt_calc = motor.variate_ambient_temp(ambientTemp_op)
    calculation_str += txt_calc
    calculation_str_print += txt_calc

    # Change U/f: Increase/decrease frequency and voltage by the same factor
    txt_calc = motor.variate_freq_volt_konstMagnFlux(Freq_op)
    calculation_str += txt_calc
    calculation_str_print += txt_calc

    # Change Voltage of operation
    txt_calc = motor.variate_voltage(Un_op)
    calculation_str += txt_calc
    calculation_str_print += txt_calc

    # Change Voltage of operation
    txt_calc = motor.variate_power(Pn_op)
    calculation_str += txt_calc
    calculation_str_print += txt_calc

    # Variate Height
    txt_calc = motor.variate_ambient_height(ambientMeter_op)
    calculation_str += txt_calc
    calculation_str_print += txt_calc

    # Recalculate In, Ma, Mk, Ia
    motor.In = motor.get_In(motor.Pn, motor.cosphi, motor.eta, motor.Un)
    motor.refresh_abs_Ia_Ma_Mn()

    # Calculate rotor parameters
    rotorVoltage_old = motor.rotorVoltage
    if motor.rotorVoltage > 0:
        # Variate the rotor connection (Y/D)
        txt_calc = motor.variate_connection_rotor(rotorChangeConnection)
        calculation_str += txt_calc
        calculation_str_print += txt_calc

        # Calculate new rotor voltage
        txt_calc = motor.variate_voltage_rotor(motor_ini.Un, motor.Un)
        calculation_str += txt_calc
        calculation_str_print += txt_calc

        # Calculate rotor current
        txt_calc = motor.update_rotor_current()
        calculation_str += txt_calc
        calculation_str_print += txt_calc
    else:
        motor.rotorVoltage = 0
        motor.rotorCurrent = 0
        motor.rotorConnection = ''

    # Create Plot for starting curves 
    fig_plt = plot_asm_start_curves([motor_ini, motor], plt_show=False)
    
    # Create pd.Dataframe of percentual changes between Initial and Operating State
    change_df = calculate_percentual_changes(motor_ini, motor)

    # Create result variable os pd.Dataframe in the format that the streamlit result variable expects
    df_result = pd.DataFrame({
        "Name": ["Pn [kW]", "Un [V]", "Freq [Hz]", "Ambient Temp. [°C]", "Height (m.a.s.l.) [m]", "Connection Y/D", "Parallel Branches (Stator)", "Ia/In [%]", "Ma/Mn [%]", "Mk/Mn [%]", "η [%]", "cos(φ)", "Nominal Speed [RPM]", "Temp. Rise (~I^2) [K]", "Temp. Rise (~I) [K]"],
        "Value": [motor.Pn, motor.Un, motor.Freq, motor.ambientTemp, motor.ambientMeter, motor.connection, motor.no_parallel, motor.Ia, motor.Ma, motor.Mk, motor.eta, motor.cosphi, motor.n, motor.deltaT_q, motor.deltaT_l],
    })

    # Result rotor df
    result_rotor_df = pd.DataFrame({
        "Name": ["Un Rotor [V]", "In Rotor [A]", "Rotor Connection"],
        "Value": [str(motor.rotorVoltage), str(motor.rotorCurrent), motor.rotorConnection]
    })

    # Print Results
    calculation_str += '\n\n' + print_header("Results") + motor.print_motor_values(show_print=False)
    if rotorVoltage > 0:
        calculation_str += '\n\n' + print_header("Rotor Parameters") + f'\nInitial Rotor Voltage: {rotorVoltage_old} V\n\n' + tabulate(result_rotor_df, headers=['Index', 'Parameter', 'Value'], tablefmt='grid')
    calculation_str += '\n\n' + print_header("Percentual Changes") + tabulate(change_df, headers=['Index', 'Parameter', 'Percentual Change [%]'], tablefmt='grid')

    return df_result, result_rotor_df, calculation_str, calculation_str_print, fig_plt, change_df

