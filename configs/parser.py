from typing import Union
from omegaconf import OmegaConf, DictConfig, ListConfig
import pandas as pd

def parse_cfg(agent_cfg_path: str, env_cfg_path: str, env_name: str) -> Union[DictConfig, ListConfig]:
    """
    Parses agent and env config files, adds c02 data and returns OmegaConf object
    """

    base = OmegaConf.load(agent_cfg_path)
    env = OmegaConf.load(env_cfg_path)

    if env_name == "MixedUseFanFCU-v0":
        base.merge_with(env.MixedUse)

        # cO2 data
        cO2_path = 'configs/c02_data/greece_2017_c02_intensity_15min.pkl'
        cO2_data = pd.read_pickle(cO2_path)
        base.cO2_year = 2017
        base.cO2_dt_col = 'datetime'
        base.cO2_carbon_col = 'carbon_intensity_avg'
        base.cO2_low = float(cO2_data[base.cO2_carbon_col].sort_values(ascending=True).values[0])
        base.cO2_high = float(cO2_data[base.cO2_carbon_col].sort_values(ascending=True).values[-1])

    elif env_name == "SeminarcenterThermostat-v0":
        base.merge_with(env.SeminarcenterThermal)

    elif env_name == 'OfficesThermostat-v0':
        base.merge_with(env.Offices)

        # cO2 data
        cO2_path = 'configs/c02_data/greece_2017_c02_intensity_15min.pkl'
        cO2_data = pd.read_pickle(cO2_path)
        base.cO2_year = 2017
        base.cO2_dt_col = 'datetime'
        base.cO2_carbon_col = 'carbon_intensity_avg'
        base.cO2_low = cO2_data[base.cO2_carbon_col].sort_values(ascending=True).values[0]
        base.cO2_high = cO2_data[base.cO2_carbon_col].sort_values(ascending=True).values[-1]

    return base

