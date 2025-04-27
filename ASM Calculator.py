# ##########################################################################################################################
# Import libraries and functions
# ##########################################################################################################################

import copy
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import tkinter as tk
from tkinter import filedialog
import io

from Functions import *

# ##########################################################################################################################
# Define relevant functions
# ##########################################################################################################################

# Copy initial values to operating values
def copy_values(edited_initial_values: pd.DataFrame):
    '''Copy initial values (left streamlit input table) to operating values (middle streamlit input table)'''
    st.session_state.initial_values = edited_initial_values.copy()
    st.session_state.operating_values['Value'] = st.session_state.operating_values['Name'].map(
        st.session_state.initial_values.set_index('Name')['Value']
    )

# Format input pd.DataFrame (containing str) as numeric pd.Dataframe (containing float for numeric values, str for Connection and None for empty/wrong values) 
def format_df_numeric(df: pd.DataFrame, error_txt_ini: str) -> tuple[pd.DataFrame, str,]:
    ''' 
    Format Input Dataframe, depending on the row: positive numeric, string\n
    Check if the input values are plausible (e.g. cos_phi can not be >1)\n
    Set all wrong values to <None>
    '''
    error_txt = error_txt_ini
    for index, row in df.iterrows():
        # Convert Table Input to the correct format: Numeric for all appart from Connection Y/D). All wrong values are converted to <None>
        if df.loc[index, 'Name'] == 'Connection Y/D':
            if df.loc[index, 'Value'].upper() != 'Y' and df.loc[index, 'Value'].upper() != 'D':
                df.loc[index, 'Value'] = None
            else:
                df.loc[index, 'Value'] = df.loc[index, 'Value'].upper()
        elif df.loc[index, 'Value'] == '':
            df.loc[index, 'Value'] = None
        else:
            # Convert to numeric value
            val_num = extract_numeric(df.loc[index, 'Value'])
            if val_num is not None:
                df.loc[index, 'Value'] = abs(val_num) # Make absolute, since no negative values make sense

        # Check if freq is in the correct range (a pole extraction is possible)
        if df.loc[index, 'Name'] == "Nominal Speed [RPM]" and df.loc[index, 'Value'] is not None:
            try:
                get_n_synchrone(n=df.loc[index, 'Value'], freq=df[df["Name"] == "Freq [Hz]"].iloc[0]["Value"] )
            except:
                df.loc[index, 'Value'] = None
        
        # Check if eta is in the correct range [1 - 100]
        if df.loc[index, 'Name'] == "η [%]" and df.loc[index, 'Value'] is not None:
            if df.loc[index, 'Value'] == '' or df.loc[index, 'Value'] < 1 or df.loc[index, 'Value'] >= 100: # eta < 100 and eta > 1
                df.loc[index, 'Value'] = None

        # Check if cos_phi is in correct range [0.1 - 0.99]
        if df.loc[index, 'Name'] == "cos(φ)" and df.loc[index, 'Value'] is not None:
            if df.loc[index, 'Value'] == '' or df.loc[index, 'Value'] < 0.1 or df.loc[index, 'Value'] >= 0.99: # cos_phi < 1 and cos_phi > 0.1
                df.loc[index, 'Value'] = None

        # Update error text, if any values where wrong (<None>)
        if df.loc[index, 'Value'] == None:
            error_txt += df.loc[index, 'Name'] + ', '
    return df, error_txt

# Calculate Bottom
def calculate_btm(edited_initial_values, edited_operating_values, values_format, rotor_voltage, change_connection_rotor):
    '''Button logic for triggering the calculation'''
    # Create editable variables
    initial_df = edited_initial_values.copy()
    operating_df = edited_operating_values.copy()
    result_df = initial_df.copy()

    # Initialize the Error text to be empty
    st.session_state.error_print = ""
    
    # Iterate over the input data and convert it to the correct format
    tables_txt = ['Machine', 'Operating']
    make_calculation = True
    for idx, df in enumerate([initial_df, operating_df]):
        error_txt_ini=f'Wrong {tables_txt[idx]} Inputs: '
        df, error_txt = format_df_numeric(df, error_txt_ini)
        
        # If error exists: Update error message and don't apply calculation
        if error_txt != error_txt_ini:
            st.session_state.error_print += '\n\n:red[' + error_txt[:-2] + ']' # In red color, with line brake and without the last ', '
            make_calculation = False

    # Calculate Results - Only of no input errors
    if make_calculation:
        # Calculate results
        result_df, result_rotor_df, calc_str_save, calc_str_print, plt_fig, change_percent = calculate_operating_values(
            # Initial machine values
            Pn = initial_df.loc[0, "Value"],
            Un = initial_df.loc[1, "Value"],
            Freq = initial_df.loc[2, "Value"], 
            ambientTemp = initial_df.loc[3, "Value"], 
            ambientMeter = initial_df.loc[4, "Value"], 
            connection = initial_df.loc[5, "Value"],
            no_parallel = initial_df.loc[6, "Value"], 
            Ia = initial_df.loc[7, "Value"],
            Ma = initial_df.loc[8, "Value"],
            Mk = initial_df.loc[9, "Value"], 
            eta = initial_df.loc[10, "Value"],
            cosphi = initial_df.loc[11, "Value"],
            n = initial_df.loc[12, "Value"],
            deltaT = initial_df.loc[13, "Value"],
            rotorVoltage = rotor_voltage,
            rotorChangeConnection = change_connection_rotor,
            motor_label_ini = 'Machine',

            # Operating values
            Pn_op = operating_df.loc[0, "Value"],
            Un_op = operating_df.loc[1, "Value"],
            Freq_op = operating_df.loc[2, "Value"], 
            ambientTemp_op = operating_df.loc[3, "Value"], 
            ambientMeter_op = operating_df.loc[4, "Value"], 
            connection_op = operating_df.loc[5, "Value"],
            no_parallel_op = operating_df.loc[6, "Value"], 
            motor_label_op = 'Operating'
        )    
    else:
        # Empty result df
        for index, row in result_df.iterrows():
            result_df.loc[index, 'Value'] = ''

        # Update text error
        st.session_state.error_print += '\n\n:red[' + 'Calculation not conducted' + ']'
        calc_str_save = None
        calc_str_print = None
        change_percent = None

    # Iterate over table dataframes, format them correctly for print
    for df in [initial_df, operating_df, result_df]:
        for idx_p, val_p in df.iterrows():
            if df.loc[idx_p, 'Value'] == None or df.loc[idx_p, 'Value'] == '':
                df.loc[idx_p, 'Value'] = ''
            elif values_format[idx_p] == 'txt': # df.loc[idx_p, 'Format']
                df.loc[idx_p, 'Value'] = df.loc[idx_p, 'Value'].upper()
            elif values_format[idx_p] in ['0']:
                df.loc[idx_p, 'Value'] = str(int(round(df.loc[idx_p, 'Value'])))
            elif values_format[idx_p] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                df.loc[idx_p, 'Value'] = str(round(df.loc[idx_p, 'Value'], ndigits=int(values_format[idx_p])))
            else:
                raise ValueError(f"Error: function: calculate_btm() --> Wrong format: {values_format[idx_p]} | Row: {val_p}")

    # Save the input data in the corresponding table
    st.session_state.initial_values = initial_df.astype(str).copy()
    st.session_state.operating_values = operating_df.astype(str).copy()
    st.session_state.result_values = result_df.astype(str).copy()
    if make_calculation:
        st.session_state.calc_print = calc_str_print
        st.session_state.calc_print_save = calc_str_save
        st.session_state.calc_plot = plt_fig
        st.session_state.change_percent = change_percent
        st.session_state.result_values_rotor = result_rotor_df


# ##########################################################################################################################
# Define relevant values
# ##########################################################################################################################

# Define the data frames
if "initial_values" not in st.session_state:
    st.session_state.initial_values = pd.DataFrame({
        "Name": ["Pn [kW]", "Un [V]", "Freq [Hz]", "Ambient Temp. [°C]", "Height (m.a.s.l.) [m]", "Connection Y/D", "Parallel Branches (Stator)", "Ia/In [%]", "Ma/Mn [%]", "Mk/Mn [%]", "η [%]", "cos(φ)", "Nominal Speed [RPM]", "Temp. Rise [K]"],
        "Value": ["", "", "50", "40", "1000", "Y", "1", "", "", "", "", "", "", "80"],
    })
if "operating_values" not in st.session_state:
    st.session_state.operating_values = pd.DataFrame({
        "Name": ["Pn [kW]", "Un [V]", "Freq [Hz]", "Ambient Temp. [°C]", "Height (m.a.s.l.) [m]", "Connection Y/D", "Parallel Branches (Stator)"],
        "Value": ["", "", "50", "40", "1000", "Y", "1"],
    })
if "result_values" not in st.session_state:
    st.session_state.result_values = st.session_state.initial_values.copy()  
# Format of the elements of column "Values" in variables st.session_state.initial_values, st.session_state.operating_values, st.session_state.result_values
values_format = ["1", '0', '0', '0', '0', "txt", '0', "1", '0', '0', '2', "2", "0", '0', '0'] # "X"=X_Decimals, '0'=Int, "txt"=Text

# Define Variables for Slip Ring
if "initial_voltage_rotor" not in st.session_state:
    st.session_state.initial_voltage_rotor = None
if "change_connection_rotor" not in st.session_state:
    st.session_state.change_connection_rotor = 'Do not change'
if "result_values_rotor" not in st.session_state:
    st.session_state.result_values_rotor = pd.DataFrame({
        "Name": ["Un Rotor [V]", "In Rotor [A]", "Rotor Connection"],
        "Value": ["", "", ""]
    })

# Define static variables
if "error_print" not in st.session_state: # Define String for errors
    st.session_state.error_print = ""
if "calc_print" not in st.session_state: # Define String for calculation text
    st.session_state.calc_print = ""
if "calc_print_save" not in st.session_state: # Define String for calculation text
    st.session_state.calc_print_save = ""
if "calc_plot" not in st.session_state: # # Define figure for plot
    st.session_state.calc_plot = None
if "change_percent" not in st.session_state: # Define Dataframe for storing Percential Change of the values by calculation
    st.session_state.change_percent = pd.DataFrame({
        "Variable": ["Pn", "B (~U/f)", "Un (per branch)", "In (per branch)", "Temp. Rise (~I^2)", "Temp. Rise (~I)"],
        "Change": [None, None, None, None, None, None],
    })

# Cyclical variables Variables
initial_values = st.session_state.initial_values.copy()
operating_values = st.session_state.operating_values.copy()
result_values = st.session_state.result_values.copy()
change_percent = st.session_state.change_percent.copy()
result_values_rotor = st.session_state.result_values_rotor.copy()

# Define containers
header_container = st.container()
input_container = st.container()


# ##########################################################################################################################
# Build web site
# ##########################################################################################################################

with header_container:
    st.markdown("<h1 style='text-align: center;'>Asynchronous Machine</h1>", unsafe_allow_html=True)
    
with input_container:
    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Table 1: On the left, Initial values
    with col1:
        st.subheader("Machine")

        row_height = 38
        height = len(st.session_state.initial_values['Value']) * row_height

        edited_initial_values = st.data_editor(
            initial_values,
            column_config={
                "Name": st.column_config.TextColumn('Variable', disabled=True),
                "Value": st.column_config.TextColumn("", disabled=False)
            },
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            height=height,
            key='initial values'
        )

    # Table 2: On the middle, Operating values
    with col2:
        st.subheader("Operating")

        row_height = 38
        height = len(st.session_state.operating_values) * row_height + int(row_height/2)

        edited_operating_values = st.data_editor(
            operating_values,
            column_config={
                "Name": st.column_config.TextColumn('Variable', disabled=True),
                "Value": st.column_config.TextColumn("", disabled=False)
            },
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            height=height,
            key="operating values"
        )

        # Button Copy values from machine values
        st.button("Copy machine values", on_click=copy_values, args=(edited_initial_values,), use_container_width=True)

        # Print Errors
        st.markdown(f'''{st.session_state.error_print}''')

    # Table 3: On the right, Result Values
    with col3:
        st.subheader("Result")

        row_height = 38
        height = len(st.session_state.result_values) * row_height #+ int(row_height/2)

        edited_result = st.data_editor(
            result_values,
            column_config={
                "Name": st.column_config.TextColumn('Variable', disabled=True),
                "Value": st.column_config.TextColumn("", disabled=True)
            },
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            height=height,
            key="result values"
        )
    
    # Slip Ring Expander
    with st.expander("Slip Ring Parameters", expanded=False):
        colex_1, colex_2, colex_3 = st.columns(3)
        with colex_1:
            Un_rotor = st.number_input("Un Rotor [V] (Initial Machine)", min_value=0, max_value=50000, step=1)
        with colex_2:
            change_connection_rotor = st.radio(
            "Change Rotor Connection Y/D:",
            ["Do not change", "Y --> D", "D --> Y"]
        )
        with colex_3:
            edited_result_values_rotor = st.data_editor(result_values_rotor, 
                                                        column_config={
                                                                "Name": st.column_config.TextColumn('Variable (Result)', disabled=True),
                                                                "Value": st.column_config.TextColumn("", disabled=True)
                                                            },
                                                        hide_index=True,)
    
    # Button Calculate variations
    st.button("Calculate", on_click=calculate_btm, args=(edited_initial_values, edited_operating_values, values_format, Un_rotor, change_connection_rotor,), use_container_width=True, type='primary')

    # Create three columns
    col2_1, col2_2= st.columns(2)

    # Save Buttons
    if st.session_state.calc_print_save is not None and st.session_state.calc_print_save != '' and st.session_state.calc_plot is not None:
        # Download Calculations.txt
        st.download_button('Download Calculation (.txt)', data=st.session_state.calc_print_save, file_name="calculations.txt")
        
        # Download Plot Immage
        # Save figure to a BytesIO buffer
        buf = io.BytesIO()
        st.session_state.calc_plot.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)  # Reset buffer position to the start
        st.download_button('Download Plot (.png)', data=buf, file_name="starting_curves.png")
        
    # Show plot
    if st.session_state.calc_plot is not None:
        st.pyplot(st.session_state.calc_plot)

    # Create three columns
    col3_1, col3_2= st.columns([1, 2])

    # Print Calculations conducted
    with col3_2:
        # Show Calculations text
        if st.session_state.calc_print is not None and st.session_state.calc_print != '':
            st.subheader("Calculations")
            st.text(st.session_state.calc_print)

    # Print Percentual Changes
    with col3_1:
        if st.session_state.change_percent.iloc[0, 1] is not None:
            st.subheader("Percentual Changes")
            edited_change = st.data_editor(
                change_percent,
                column_config={
                    "Variable": st.column_config.TextColumn('Parameter', disabled=True),
                    "Change": st.column_config.TextColumn("Change [%]", disabled=True)
                },
                num_rows="fixed",
                hide_index=True,
                use_container_width=True,
                key="change values"
            )

