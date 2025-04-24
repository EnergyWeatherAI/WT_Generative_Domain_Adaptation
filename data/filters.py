import numpy as np 
import scipy.linalg
from scipy.spatial import distance

###
#
# Given a dataframe of SCADA data, these functions set a filter flag for excluding outliers.
# Procedure is mainly based on the description in McKinnon et al., 2022 (iopscience.iop.org/article/10.1088/1742-6596/2151/1/012005) to which we refer for technical details.
#
###


def rated_filter(wt_df, wind_speed_col_name, power_col_name, rated_windspeed, rated_power):
    '''
    Sets a dataframe column rated_filter to 1.0 to possibly adjust for curtailments and power-related outliers.

    Args:
        wt_df (pandas dataframe): the loaded dataframe of the SCADA data (see csv_loader)
        wind_speed_col_name (str): The column name of the wind speed for which the filter should be applied to (typically: WindSpeed_avg)
        power_col_name (str): The column name of the power for which the filter should be applied to (typically: Power_avg) 
        rated_windspeed (float): The rated wind speed of the corresponding WT in m/s (or system equivalent)
        rated_power (float): The rated power of the corresponding WT in KW (or system equivalent)

    The rated filter is mainly used to adjust for curtailments:
    Sets a filter flag (rated_filter == 1.0) for all values where the windspeed was above the rated windspeed (m/s)
    but with a power output less than 97.5% of the rated power (KW).  
    '''

    wt_df["rated_filter"] = 0
    rated_idxs = wt_df[(wt_df[wind_speed_col_name] >= rated_windspeed) & (wt_df[power_col_name] < 0.975 * rated_power)].index 
    wt_df.loc[rated_idxs, "rated_filter"] = 1.0
    return wt_df


def set_mahal_distance(wt_df, wind_speed_col_name, power_col_name):
    '''
    Sets a filter value flag using Mahalanobis distance and binning (see paper).

    Args:
        wt_df (pandas dataframe): the loaded dataframe of the SCADA data (see csv_loader)
        wind_speed_col_name (str): The column name of the wind speed for which the filter should be applied to (typically: WindSpeed_avg)
        power_col_name (str): The column name of the power for which the filter should be applied to (typically: Power_avg) 
    
    The m_dist column in the dataframe can be subsequently used to exclude these values, based on e.g., a value of 2.0 and higher.
    '''

    wt_df["m_dist"] = np.nan

    # set wind speed bins
    ws_starts = [0] + list(np.arange(1, int(np.max(wt_df[wind_speed_col_name].values))-2, 0.5))
    ws_ends =   [2.6] + [ws + 1.0 for ws in ws_starts[1:]]
    ws_ends[-1] = ws_ends[-1] + 2.0

    # calculate distance for each bin seperately
    for start, end in zip(ws_starts, ws_ends):
        bin_df = wt_df[(wt_df[wind_speed_col_name] >= start) & (wt_df[wind_speed_col_name] <= end)]
        bin_pwr_median, bin_ws_median = np.median(bin_df[power_col_name].values), np.median(bin_df[wind_speed_col_name].values)

        # distance calculations
        cluster_median = np.array([bin_pwr_median, bin_ws_median])
        X = np.dstack([bin_df[power_col_name].values, bin_df[wind_speed_col_name].values])[0]
        if X.shape[0] > 3:
            covariance_matrix = scipy.linalg.inv(np.cov(X.T))
            distances = []
            for val in X:
                distances.append(distance.mahalanobis(val, cluster_median, covariance_matrix))
        
        else:
            distances = np.zeros(X.shape[0], )

        bin_idxs = wt_df[(wt_df[wind_speed_col_name] >= start) & (wt_df[wind_speed_col_name] <= end)].index
        wt_df.loc[bin_idxs, "m_dist"] = distances
        
    return wt_df
