import pandas as pd 
import data.filters

###
# Functions within this file are used to convert SCADA csv files with measurements and incident flags (see dummy dataset)
# into pandas dataframes. At the same time, a normal data filter can be applied to exclude outliers from the training set (see filters.py).
# Some domain-specific manual cleaning is included; might need to be adapted to specific user cases.  
###

def load_csv_as_df(path, x_features, rated_ws, rated_pwr, filter_incidents = True, keep_incident_flag = False):
    '''
    Loads a csv SCADA file specified by the path into a dataframe.

    Args:
        x_features (list of str): Only SCADA features specified in the x_features list are kept for the dataframe.
        rated_ws (float): rated wind speed of the respective WT in m/s
        rated_pwr (float): rated power of the respective WT in KW (or custom SCADA-system equivalent)
        filter_incidents (bool): if set to True, incidents (based on column) and outliers (based on filtering) will be excluded. Set to False for evaluation data.
        keep_incident_flag (bool): if set to True, the dataframe will retain the incident column additionally to the x_features columns
    '''

    wt_df = pd.read_csv(path, parse_dates=["Datetime"], date_format = "%Y.%m.%d %H:%M:%S") # see example dataset
    wt_df = wt_df.iloc[:-1]  # remove last line, needed for some files
    wt_df = rename_main_columns(wt_df) # renaming changing SCADA variable names for consistent df column names

    # INCIDENT FILTER
    # (rows with incident != 0) specified in the csv:
    if filter_incidents:
        incident_indices = wt_df[wt_df.incident > 0.0].index 
        wt_df.drop(incident_indices, inplace=True)

    # slight manual measurement cleaning not caught within the filter
    wt_df = clean_sensor_errors(wt_df, x_features) # will have to be adjusted for specific user cases

    # DATA FILTERING
    # underperformance rated power filter
    if filter_incidents:
        wt_df = data.filters.rated_filter(wt_df, "WindSpeed_avg", "Power_avg", rated_ws, rated_pwr)
        rated_filtered = wt_df[wt_df.rated_filter == 1.0].index 
        wt_df.drop(rated_filtered, inplace=True)

    # mahal. filter for outliers and remainder
    if filter_incidents:
        wt_df = data.filters.set_mahal_distance(wt_df, "WindSpeed_avg", "Power_avg")
        maha_filtered = wt_df[wt_df.m_dist > 2.0].index 
        wt_df.drop(maha_filtered, inplace=True)

    # create a dataframe with a continuous timescale
    wt_df["Timestamp"] = pd.to_datetime(wt_df["Timestamp"], utc=True)
    # create a sequential time to be able to extract timesequences
    start_date = wt_df.iloc[0].loc["Timestamp"]
    end_date = wt_df.iloc[-1].loc["Timestamp"]
    daterange = pd.date_range(start=start_date, end=end_date, freq="10min") # NOTE assumption: 10-minute SCADA data
    df_t = pd.DataFrame() 
    df_t["Timestamp"] = daterange 
    df = df_t.merge(wt_df, how='left', on="Timestamp")
    df = df.interpolate(method ='linear', limit=4, axis=0, limit_direction='forward') # interpolate a maximum of 4 steps

    # only keep the wanted columns (time, x features)
    wanted_columns = ["Timestamp"] + x_features 
    if keep_incident_flag: wanted_columns = wanted_columns + ["incident"]

    df = df[wanted_columns]
    return df



def clean_sensor_errors(wt_df, x_features):
    for pwr_var in ["Power_min", "Power_avg", "Power_max"]:
        if pwr_var in x_features:
            neg_powers = wt_df[wt_df[pwr_var] < 0].index
            wt_df.loc[neg_powers, pwr_var] = 0

    for ws_var in ["WindSpeed_min", "WindSpeed_avg", "WindSpeed_max"]:
        if ws_var in x_features:
            over_ws = wt_df[wt_df[ws_var] > 40].index
            wt_df.drop(over_ws, inplace=True)

    for rotor_var in ["RotorSpeed_min", "RotorSpeed_avg", "RotorSpeed_max"]:
        if rotor_var in x_features:
            over_rtr = wt_df[wt_df[rotor_var] > 30].index
            wt_df.drop(over_rtr, inplace=True)

    for temp_var in ["RotorTemp1", "StatorTemp1", "FrontBearingTemp", "RearBearingTemp"]:
        if temp_var in x_features:
            odd_temp = wt_df[(wt_df[temp_var] > 150) | (wt_df[temp_var] < 0)].index
            wt_df.drop(odd_temp, inplace=True)

    return wt_df


def rename_main_columns(df):
    df.rename(columns={"Datetime": "Timestamp"}, inplace=True)

    if "Power_avg" not in df.columns:
        if "GrdProdPower_avg" in df.columns:
            df.rename(columns={"GrdProdPower_min": "Power_min",
                                "GrdProdPower_avg": "Power_avg", 
                                    "GrdProdPower_max": "Power_max"}, inplace=True)

    if "RotorSpeed_avg" not in df.columns:
        if "RotorRPM_min" in df.columns:
            df.rename(columns={"RotorRPM_min": "RotorSpeed_min",
                                "RotorRPM_avg": "RotorSpeed_avg", 
                                    "RotorRPM_max": "RotorSpeed_max"}, inplace=True)

    return df
