import pandas as pd
import os


class MixedUse:
    def __init__(self):
        self.redundant_observations = [
            'time',
            'Bd_Pw_All',  # building power consumption
            'Bd_Pw_chiller',  # chiller power consumption
            'Bd_Pw_coil',  # heating coil power consumption
            'Bd_Pw_cooling_AHU2',  # AHU 2 cooling power
            'Bd_Pw_cooling_HP',  # heat pump cooling power
            'Bd_Pw_heating_HP',  # heat pump heating power
            'Fa_Pw_HVAC',  # hvac power consumption
        ]

        self.redundant_actions = []  # no redundant actions for this environment

        self.discrete_actions = []  # no discrete actions for this environment

        self.continuous_actions = [
            'Z02_T_Thermostat_sp',
            'Z03_T_Thermostat_sp',
            'Z04_T_Thermostat_sp',
            'Z05_T_Thermostat_sp',
            'Z08_T_Thermostat_sp',
            'Z09_T_Thermostat_sp',
            'Z10_T_Thermostat_sp',
            'Z11_T_Thermostat_sp',
            'Bd_Fl_AHU1_sp',
            'Bd_Fl_AHU2_sp',
            'Bd_T_AHU1_sp',
            'Bd_T_AHU2_sp'
        ]

        self.temp_reward = [
            'Z02_T',  # zone 2 temperature
            'Z03_T',  # zone 3 temperature
            'Z04_T',  # zone 4 temperature
            'Z05_T',  # zone 5 temperature
            'Z08_T',  # zone 8 temperature
            'Z09_T',  # zone 9 temperature
            'Z10_T',  # zone 10 temperature
            'Z11_T',  # zone 11 temperature
        ]

        # energy related features
        self.energy_reward = [
            'Fa_Pw_All',  # total power consumption
        ]

        self.c02_reward = [
            'c02'
        ]

        self.disturbances = [
            'Ext_Irr',  # irradiance
            'Ext_P',  # pressure
            'Ext_RH',  # relative humidity
            'Ext_T',  # temperature
        ]

        self.lower_temp_goal = 19
        self.upper_temp_goal = 24

        self.c02_path = os.path.join(os.getcwd(), 'config/c02_data/greece_2017_c02_intensity_15min.pkl')
        self.c02_data = pd.read_pickle(self.c02_path)
        self.c02_year = 2017
        self.c02_dt_col = 'datetime'
        self.c02_carbon_col = 'carbon_intensity_avg'
        self.c02_low = self.c02_data[self.c02_carbon_col].sort_values(ascending=True).values[0]
        self.c02_high = self.c02_data[self.c02_carbon_col].sort_values(ascending=True).values[-1]

        self.weather = "GRC_A_Athens"


class Offices:
    def __init__(self):
        self.redundant_observations = [
            'time',
            'Bd_Pw_All',  # building power consumption
            'Fa_Pw_PV',  # PV power production
            'Fa_Pw_HVAC',  # hvac power consumption
        ]

        self.redundant_actions = []  # no redundant actions for this environment

        self.discrete_actions = [
            'Bd_Cooling_onoff_sp',
            'Bd_Heating_onoff_sp'
        ]

        self.continuous_actions = [
            'Z01_T_Thermostat_sp',
            'Z02_T_Thermostat_sp',
            'Z03_T_Thermostat_sp',
            'Z04_T_Thermostat_sp',
            'Z05_T_Thermostat_sp',
            'Z06_T_Thermostat_sp',
            'Z07_T_Thermostat_sp',
            'Z15_T_Thermostat_sp',
            'Z16_T_Thermostat_sp',
            'Z17_T_Thermostat_sp',
            'Z18_T_Thermostat_sp',
            'Z19_T_Thermostat_sp',
            'Z20_T_Thermostat_sp',
            'Z25_T_Thermostat_sp',
        ]

        self.temp_reward = [
            'Z01_T',
            'Z02_T',
            'Z03_T',
            'Z04_T',
            'Z05_T',
            'Z06_T',
            'Z07_T',
            'Z15_T',
            'Z16_T',
            'Z17_T',
            'Z18_T',
            'Z19_T',
            'Z20_T',
            'Z25_T',
        ]

        # energy related features
        self.energy_reward = [
            'Fa_Pw_All',  # total power consumption
        ]

        self.c02_reward = [
            'c02'
        ]

        self.disturbances = [
            'Ext_Irr',  # irradiance
            'Ext_P',  # pressure
            'Ext_RH',  # relative humidity
            'Ext_T',  # temperature
        ]

        self.lower_temp_goal = 19
        self.upper_temp_goal = 24

        self.c02_path = os.path.join(os.getcwd(), 'config/c02_data/greece_2017_c02_intensity_15min.pkl')
        self.c02_data = pd.read_pickle(self.c02_path)
        self.c02_year = 2017
        self.c02_dt_col = 'datetime'
        self.c02_carbon_col = 'carbon_intensity_avg'
        self.c02_low = self.c02_data[self.c02_carbon_col].sort_values(ascending=True).values[0]
        self.c02_high = self.c02_data[self.c02_carbon_col].sort_values(ascending=True).values[-1]

        self.weather = "GRC_A_Athens"


class Apartments2Thermal:
    def __init__(self):
        self.redundant_observations = [
            'time',
            'Bd_Pw_Bat_sp_out',  # battery charging setpoint
            'Bd_Ch_EV1Bat_sp_out',  # EV1 charging setpoint
            'Bd_Ch_EV2Bat_sp_out',  # EV2 charging setpoint
            'Bd_DisCh_EV1Bat',  # EV1 discharging rate
            'Bd_DisCh_EV2Bat',  # EV2 discharging rate
            'Z01_E_Appl',  # Z01 appliance energy
            'Z02_E_Appl',  # Z01 appliance energy
            'Z03_E_Appl',  # Z01 appliance energy
            'Z04_E_Appl',  # Z01 appliance energy
            'Z05_E_Appl',  # Z01 appliance energy
            'Z06_E_Appl',  # Z01 appliance energy
            'Z07_E_Appl',  # Z01 appliance energy
            'Z08_E_Appl',  # Z01 appliance energy
            'Fa_Stat_EV1',  # EV1 availability
            'Fa_ECh_EV1Bat',  # EV1 battery charging energy
            'Fa_EDCh_EV1Bat',  # EV1 battery discharging energy
            'Fa_Stat_EV2',  # EV2 availability
            'Fa_ECh_EV2Bat',  # EV2 battery charging energy
            'Fa_EDCh_EV2Bat',  # EV2 battery discharging energy
            'Fa_ECh_Bat',  # battery charging energy
            'Fa_EDCh_Bat',  # battery discharging energy
            'Bd_FracCh_EV1Bat',  # EV1 battery state of charge
            'Bd_FracCh_EV2Bat',  # EV2 battery state of charge
            'Bd_FracCh_Bat',  # battery state of charge
            'Fa_Pw_All',  # total power consumption
            'Fa_Pw_Prod',  # PV power production
            'Fa_E_self',  # Self consumption energy
            'Fa_E_All',  # total energy consumption
            'Fa_E_Light',  # lighing energy
            'Fa_E_Appl',  # appliances energy
        ]

        self.redundant_actions = [
            'Bd_Pw_Bat_sp',
            'Bd_Ch_EV1Bat_sp',
            'Bd_Ch_EV2Bat_sp',
            'Bd_DisCh_EV1Bat_sp'
        ]

        self.discrete_actions = [
            'P1_onoff_HP_sp',
            'P2_onoff_HP_sp',
            'P3_onoff_HP_sp',
            'P4_onoff_HP_sp'
        ]

        self.continuous_actions = [
            'P1_T_Thermostat_sp',
            'P2_T_Thermostat_sp',
            'P3_T_Thermostat_sp',
            'P4_T_Thermostat_sp',
        ]

        self.temp_reward = [
            'Z01_T',  # zone 1 temperature
            'Z02_T',  # zone 2 temperature
            'Z03_T',  # zone 3 temperature
            'Z04_T',  # zone 4 temperature
            'Z05_T',  # zone 5 temperature
            'Z06_T',  # zone 6 temperature
            'Z07_T',  # zone 7 temperature
            'Z08_T',  # zone 8 temperature
        ]

        # energy related features
        self.energy_reward = [
            'Fa_E_HVAC',  # hvac energy consumption
        ]

        self.c02_reward = [
            'c02'
        ]

        self.disturbances = [
            'Ext_Irr',  # irradiance
            'Ext_P',  # pressure
            'Ext_RH',  # relative humidity
            'Ext_T',  # temperature
        ]

        self.lower_temp_goal = 19
        self.upper_temp_goal = 24

        self.c02_path = os.path.join(os.getcwd(), 'config/c02_data/spain_2017_c02_intensity_3min.pkl')
        self.c02_data = pd.read_pickle(self.c02_path)
        self.c02_year = 2017
        self.c02_dt_col = 'datetime'
        self.c02_carbon_col = 'carbon_intensity_avg'
        self.c02_low = self.c02_data[self.c02_carbon_col].sort_values(ascending=True).values[0]
        self.c02_high = self.c02_data[self.c02_carbon_col].sort_values(ascending=True).values[-1]

        self.weather = "ESP_CT_Barcelona"


class SeminarcenterThermal:
    def __init__(self):
        self.redundant_observations = [
            'time',
            'Bd_Pw_boiler',
            'Bd_Pw_prod',
            'Bd_T_Boiler_sp_out',
            'Bd_CO2',
            'Bd_Pw_All',
            'Fa_Pw_Pur',
            'Fa_Pw_HVAC',
        ]

        self.redundant_actions = [
            'Grid_CO2_sp'
        ]

        self.discrete_actions = [
        ]

        self.continuous_actions = [
            'Z01_T_Thermostat_sp',
            'Z02_T_Thermostat_sp',
            'Z03_T_Thermostat_sp',
            'Z04_T_Thermostat_sp',
            'Z05_T_Thermostat_sp',
            'Z06_T_Thermostat_sp',
            'Z08_T_Thermostat_sp',
            'Z09_T_Thermostat_sp',
            'Z10_T_Thermostat_sp',
            'Z11_T_Thermostat_sp',
            'Z13_T_Thermostat_sp',
            'Z14_T_Thermostat_sp',
            'Z15_T_Thermostat_sp',
            'Z18_T_Thermostat_sp',
            'Z19_T_Thermostat_sp',
            'Z20_T_Thermostat_sp',
            'Z21_T_Thermostat_sp',
            'Z22_T_Thermostat_sp'
        ]

        self.temp_reward = [
            'Z01_T',  # zone 1 temperature
            'Z02_T',  # zone 2 temperature
            'Z03_T',  # zone 3 temperature
            'Z04_T',  # zone 4 temperature
            'Z05_T',  # zone 5 temperature
            'Z06_T',  # zone 6 temperature
            'Z08_T',  # zone 8 temperature
            'Z09_T',  # zone 9 temperature
            'Z10_T',  # zone 10 temperature
            'Z11_T',  # zone 11 temperature
            'Z13_T',  # zone 13 temperature
            'Z14_T',  # zone 14 temperature
            'Z15_T',  # zone 15 temperature
            'Z18_T',  # zone 18 temperature
            'Z19_T',  # zone 19 temperature
            'Z20_T',  # zone 20 temperature
            'Z21_T',  # zone 21 temperature
            'Z22_T',  # zone 22 temperature
        ]

        # energy related features
        self.energy_reward = [
            'Fa_Pw_All',  # building power consumption
        ]

        self.c02_reward = [
            'Grid_CO2'
        ]

        self.disturbances = [
            'Ext_Irr',  # irradiance
            'Ext_P',  # pressure
            'Ext_RH',  # relative humidity
            'Ext_T',  # temperature
        ]

        self.lower_temp_goal = 19
        self.upper_temp_goal = 24

        self.weather = "DNK_MJ_Horsens1"
