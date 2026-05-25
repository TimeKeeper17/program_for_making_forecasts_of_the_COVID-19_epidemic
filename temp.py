import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime

DF_confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
DF_deaths = pd.read_csv('time_series_covid19_deaths_global.csv')
DF_recovered = pd.read_csv('time_series_covid19_recovered_global.csv')


def add_china_total_to_df(df):
    df_with_total = df.copy()
    china_rows = df[df['Country/Region'] == 'China']
    total_row = {}
    total_row['Province/State'] = np.nan
    total_row['Country/Region'] = 'China'
    total_row['Lat'] = china_rows['Lat'].mean()
    total_row['Long'] = china_rows['Long'].mean()
    date_columns = [col for col in df.columns if col not in ['Province/State', 'Country/Region', 'Lat', 'Long']]
    for col in date_columns:
        total_row[col] = china_rows[col].sum()
    df_with_total = pd.concat([df_with_total, pd.DataFrame([total_row])], ignore_index=True)
    return df_with_total


DF_confirmed = add_china_total_to_df(DF_confirmed)
DF_deaths = add_china_total_to_df(DF_deaths)
DF_recovered = add_china_total_to_df(DF_recovered)

selected_countries = [
    'China', 'US', 'India', 'Germany', 'Korea, South', 'Japan',
    'United Kingdom', 'Russia', 'South Africa', 'Nigeria',
    'Iran', 'Indonesia', 'Chile', 'Netherlands', 'Brazil'
]


def process_covid_data(df, countries):
    df_filtered = df[df['Country/Region'].isin(countries) & df['Province/State'].isna()]
    df_filtered = df_filtered.drop(['Lat', 'Long', 'Province/State'], axis=1)
    df_filtered = df_filtered.set_index('Country/Region')
    df_transposed = df_filtered.T
    df_transposed.index = pd.to_datetime(df_transposed.index)
    df_transposed = df_transposed.sort_index()
    return df_transposed


C = process_covid_data(DF_confirmed, selected_countries)
D = process_covid_data(DF_deaths, selected_countries)
R = process_covid_data(DF_recovered, selected_countries)
R_total = D + R


def get_country_table(country_name, C_df, R_total_df, start_date='2020-03-01'):
    if country_name not in C_df.columns:
        raise ValueError(f"Страна '{country_name}' не найдена в данных")
    C_series = C_df[country_name]
    R_series = R_total_df[country_name]
    df_country = pd.DataFrame({'C': C_series, 'R': R_series})
    df_country['I'] = df_country['C'] - df_country['R']
    df_country = df_country[df_country.index >= start_date]
    df_country['Date'] = df_country.index.strftime('%d.%m.%Y')
    df_country['Time'] = range(1, len(df_country) + 1)
    df_country = df_country[['Time', 'Date', 'C', 'R', 'I']]
    df_country = df_country.reset_index(drop=True)
    return df_country


def main(df, w_prev, w_next, t_start, t_end, country, output_dir, w_last_start):
    df_now = data_C(df)
    n = df_now.shape[0]
    t_start_index = df_now.index[df_now['Time'] == t_start].tolist()
    t_end_ind = int(n + t_end - df_now.Time[n - 1]) - 1
    t_start_ind = t_start_index[0]
    df_now_ = df_now.copy()
    df_now_1, df_now_2, df_now_3, df_now_4, df_now_5, df_now_6 = [df_now_.copy() for _ in range(6)]

    func_dict = {'approximate': approximate}
    func_name = 'approximate'
    par = 0

    df_pred_1 = def_pred_1(df_now_1, w_prev, w_next, t_start_ind, t_end_ind, par, func_dict, func_name, w_last_start)
    df_pred_2 = def_pred_1(df_now_2, w_prev, w_next, t_start_ind, t_end_ind, 1, func_dict, func_name, w_last_start)
    df_pred_3 = def_pred_2(df_now_3, w_prev, w_next, t_start_ind, t_end_ind, par, func_dict, func_name, w_last_start)
    df_pred_4 = def_pred_2(df_now_4, w_prev, w_next, t_start_ind, t_end_ind, 1, func_dict, func_name, w_last_start)
    df_pred_5 = def_pred_3(df_now_5, w_prev, w_next, t_start_ind, t_end_ind, par, func_dict, func_name, w_last_start)
    df_pred_6 = def_pred_3(df_now_6, w_prev, w_next, t_start_ind, t_end_ind, 1, func_dict, func_name, w_last_start)

    dfs = [df_pred_1, df_pred_2, df_pred_3, df_pred_4, df_pred_5, df_pred_6]

    df_error = pd.concat([df['error'] for df in dfs], axis=1)
    df_MAPE = pd.concat([df['MAPE'] for df in dfs], axis=1)
    df_MAPE_all = pd.concat([df['MAPE_all'] for df in dfs], axis=1)
    df_error_C = pd.concat([df['error_C'] for df in dfs], axis=1)
    df_MAPE_C = pd.concat([df['MAPE_C'] for df in dfs], axis=1)
    df_MAPE_C_all = pd.concat([df['MAPE_C_all'] for df in dfs], axis=1)

    for metric_df in [df_error, df_MAPE, df_MAPE_all, df_error_C, df_MAPE_C, df_MAPE_C_all]:
        metric_df.insert(loc=0, column='Date', value=df_pred_1.Date)
        metric_df.insert(loc=0, column='Time', value=df_pred_1.Time)

    output_path = os.path.join(output_dir, f'{country}_{w_prev}_predict.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        for i, df_i in enumerate(dfs + [df_error, df_MAPE, df_MAPE_all, df_error_C, df_MAPE_C, df_MAPE_C_all], start=1):
            df_i.to_excel(writer, sheet_name=f'Лист{i}', index=False)
    return df_error, df_MAPE, df_MAPE_all, df_error_C, df_MAPE_C, df_MAPE_C_all


def approximate(df, w_prev, w_next, par=0):
    def power_func(x, a, b):
        return a * np.power(x, b)

    def linear_func(x, a, b):
        return a + b * x

    x_data = np.arange(1, len(df) + 1)
    y_data = np.array(df).astype(float)

    if par == 0:
        x_extended = np.arange(len(df) + 1, len(df) + w_next + 1)
        k = w_next - 1
    else:
        x_extended = np.arange(len(df) + 1, len(df) + 2)
        k = 0

    ans = np.array([])
    ans_func = np.array(['-'] * w_next)

    for i in range(k, w_next):
        if np.any(y_data <= 0):
            a, b = np.polyfit(x_data, y_data, 1)[1], np.polyfit(x_data, y_data, 1)[0]
            extrapolated_values = linear_func(x_extended, a, b)
            ans_func[i] = 'linear' if k == 0 else 'linear'
        else:
            coef = np.polyfit(np.log(x_data), np.log(y_data), 1)
            coef[1] = np.exp(coef[1])
            a, b = coef[1], coef[0]
            extrapolated_values = power_func(x_extended, a, b)
            ans_func[i] = 'power' if k == 0 else 'power'
        ans = np.concatenate((ans, extrapolated_values))
        y_data = np.append(y_data[1:], extrapolated_values[0])
    return ans, ans_func


def def_pred_1(df, w_prev, w_next, t_start_ind, t_end_ind, par, func_dict, func, w_last_start):
    n_0 = df.shape[0]
    empty_data = pd.DataFrame(np.nan, index=range(t_end_ind + 1 - n_0), columns=df.columns)
    df = pd.concat([df, empty_data], ignore_index=True)
    n = df.shape[0]

    init_cols = {
        'zero': ['inflow_pred', 'outflow_pred', 'Total_pred', 'error', 'error_C', 'MAPE_all', 'MAPE_C_all'],
        'dash': ['MAPE', 'MAPE_C', 'mean_MAPE']
    }
    for col in init_cols['zero']: df[col] = [0] * n
    for col in init_cols['dash']: df[col] = ['-'] * n

    for i in range(t_start_ind, t_end_ind, w_next):
        t_prev = w_last_start if w_prev == -1 else max(0, i - w_prev + 1)
        t_next = min(i + w_next, t_end_ind)

        mask = df.inflow[t_prev:i + 1].isna()
        X1 = df.inflow[t_prev:i + 1].where(~mask, df.inflow_pred[t_prev:i + 1])
        X2 = df.outflow[t_prev:i + 1].where(~mask, df.outflow_pred[t_prev:i + 1])

        df.inflow_pred[i + 1:t_next + 1], _ = func_dict[func](X1, w_prev, t_next - i, par)
        df.outflow_pred[i + 1:t_next + 1], _ = func_dict[func](X2, w_prev, t_next - i, par)

        for j in range(i + 1, t_next + 1):
            df.Total_pred[j] = df.inflow_pred[j] - df.outflow_pred[j]
            df.error[j] = abs(df.Total_pred[j] - df.Total[j]) / df.Total[j] * 100
            df.error_C[j] = abs(df.inflow_pred[j] - df.inflow[j]) / df.inflow[j] * 100
            df.MAPE_all[j] = np.mean(df.error[i + 1:j + 1])
            df.MAPE_C_all[j] = np.mean(df.error_C[i + 1:j + 1])
        df.MAPE[t_next] = np.mean(df.error[i + 1:t_next + 1])
        df.MAPE_C[t_next] = np.mean(df.error_C[i + 1:t_next + 1])

    for i in range(t_start_ind, t_end_ind, w_next):
        df.mean_MAPE[i + w_next] = np.mean(df.error[i + 1:n_0])
    df.mean_MAPE[n_0 - 1] = df.MAPE[n_0 - 1]
    df['accuracy'] = 100 - df['error']
    df.accuracy[:n_0] = 100 - df.error[:n_0]
    return df


def def_pred_2(df, w_prev, w_next, t_start_ind, t_end_ind, par, func_dict, func, w_last_start):
    n_0 = df.shape[0]
    empty_data = pd.DataFrame(np.nan, index=range(t_end_ind + 1 - n_0), columns=df.columns)
    df = pd.concat([df, empty_data], ignore_index=True)
    n = df.shape[0]

    df['inflow_int'] = df['inflow']
    df['outflow_int'] = df['outflow']

    init_cols = {
        'zero': ['r_inflow_int', 'r_outflow_int', 'r_inflow_int_pred', 'r_outflow_int_pred',
                 'inflow_int_pred', 'outflow_int_pred', 'Total_pred', 'error', 'error_C',
                 'MAPE_all', 'MAPE_C_all'],
        'dash': ['r_checking', 'MAPE', 'MAPE_C', 'mean_MAPE']
    }
    for col in init_cols['zero']: df[col] = [0] * n
    for col in init_cols['dash']: df[col] = ['-'] * n

    for i in range(1, n):
        df.r_inflow_int[i] = df.inflow_int[i] / df.inflow_int[i - 1] - 1
        df.r_outflow_int[i] = df.outflow_int[i] / df.outflow_int[i - 1] - 1
        if df.r_inflow_int[i] <= 0 or df.r_outflow_int[i] <= 0:
            df.r_checking[i] = 'WARNING'

    if t_start_ind <= 1:
        t_start_ind += w_next

    for i in range(t_start_ind, t_end_ind, w_next):
        t_prev = w_last_start if w_prev == -1 else max(1, i - w_prev + 1)
        t_next = min(i + w_next, t_end_ind)

        mask = df.r_inflow_int[t_prev:i + 1].isna()
        X1 = df.r_inflow_int[t_prev:i + 1].where(~mask, df.r_inflow_int_pred[t_prev:i + 1])
        X2 = df.r_outflow_int[t_prev:i + 1].where(~mask, df.r_outflow_int_pred[t_prev:i + 1])

        df.r_inflow_int_pred[i + 1:t_next + 1], _ = func_dict[func](X1, w_prev, t_next - i, par)
        df.r_outflow_int_pred[i + 1:t_next + 1], _ = func_dict[func](X2, w_prev, t_next - i, par)

        if np.isnan(df.inflow_int[i]):
            df.inflow_int_pred[i + 1] = df.inflow_int_pred[i] * (df.r_inflow_int_pred[i + 1] + 1)
            df.outflow_int_pred[i + 1] = df.outflow_int_pred[i] * (df.r_outflow_int_pred[i + 1] + 1)
        else:
            df.inflow_int_pred[i + 1] = df.inflow_int[i] * (df.r_inflow_int_pred[i + 1] + 1)
            df.outflow_int_pred[i + 1] = df.outflow_int[i] * (df.r_outflow_int_pred[i + 1] + 1)

        for j in range(1, t_next - i):
            df.inflow_int_pred[i + 1 + j] = df.inflow_int_pred[i + j] * (df.r_inflow_int_pred[i + 1 + j] + 1)
            df.outflow_int_pred[i + 1 + j] = df.outflow_int_pred[i + j] * (df.r_outflow_int_pred[i + 1 + j] + 1)

        for j in range(t_next - i):
            df.Total_pred[i + 1 + j] = df.inflow_int_pred[i + 1 + j] - df.outflow_int_pred[i + 1 + j]
            df.error[i + 1 + j] = abs(df.Total_pred[i + 1 + j] - df.Total[i + 1 + j]) / df.Total[i + 1 + j] * 100
            df.error_C[i + 1 + j] = abs(df.inflow_int_pred[i + 1 + j] - df.inflow[i + 1 + j]) / df.inflow[
                i + 1 + j] * 100
            df.MAPE_all[i + 1 + j] = np.mean(df.error[i + 1:i + j + 2])
            df.MAPE_C_all[i + 1 + j] = np.mean(df.error_C[i + 1:i + j + 2])
        df.MAPE[t_next] = np.mean(df.error[i + 1:t_next + 1])
        df.MAPE_C[t_next] = np.mean(df.error_C[i + 1:t_next + 1])

    for i in range(t_start_ind, t_end_ind, w_next):
        df.mean_MAPE[i + w_next] = np.mean(df.error[i + 1:n_0])
    df.mean_MAPE[n_0 - 1] = df.MAPE[n_0 - 1]
    df['accuracy'] = 100 - df['error']
    df.accuracy[:n_0] = 100 - df.error[:n_0]
    return df


def def_pred_3(df, w_prev, w_next, t_start_ind, t_end_ind, par, func_dict, func, w_last_start):
    n_0 = df.shape[0]
    empty_data = pd.DataFrame(np.nan, index=range(t_end_ind + 1 - n_0), columns=df.columns)
    df = pd.concat([df, empty_data], ignore_index=True)
    n = df.shape[0]

    df['inflow_int'] = df['inflow']
    df['outflow_int'] = df['outflow']

    init_cols_zero = ['r_inflow_int', 'r_outflow_int', 'alpha_r_inflow_int', 'alpha_r_outflow_int',
                      'alpha_r_inflow_int_pred', 'alpha_r_outflow_int_pred', 'r_inflow_int_pred',
                      'r_outflow_int_pred', 'inflow_int_pred', 'outflow_int_pred', 'Total_pred',
                      'error', 'error_C', 'MAPE_all', 'MAPE_C_all']
    init_cols_dash = ['r_checking', 'used_func_1', 'used_func_2', 'MAPE', 'MAPE_C', 'mean_MAPE']

    for col in init_cols_zero: df[col] = [0] * n
    for col in init_cols_dash: df[col] = ['-'] * n

    for i in range(1, n):
        df.r_inflow_int[i] = df.inflow_int[i] / df.inflow_int[i - 1] - 1
        df.r_outflow_int[i] = df.outflow_int[i] / df.outflow_int[i - 1] - 1
        if df.r_inflow_int[i] <= 0 or df.r_outflow_int[i] <= 0:
            df.r_checking[i] = 'WARNING'

    for i in range(1 + w_next, n):
        df.alpha_r_inflow_int[i] = (df.r_inflow_int[i - w_next + 1] - df.r_inflow_int[i]) / w_next
        df.alpha_r_outflow_int[i] = (df.r_outflow_int[i - w_next + 1] - df.r_outflow_int[i]) / w_next

    if t_start_ind <= 1 + w_prev:
        t_start_ind += w_next

    for i in range(t_start_ind, t_end_ind, w_next):
        t_prev = w_last_start if w_prev == -1 else max(1 + w_next, i - w_prev + 1)
        t_next = min(i + w_next, t_end_ind)

        if len(df.alpha_r_inflow_int[t_prev:i + 1]) <= 1:
            continue

        mask = df.alpha_r_inflow_int[t_prev:i + 1].isna()
        X1 = df.alpha_r_inflow_int[t_prev:i + 1].where(~mask, df.alpha_r_inflow_int_pred[t_prev:i + 1])
        X2 = df.alpha_r_outflow_int[t_prev:i + 1].where(~mask, df.alpha_r_outflow_int_pred[t_prev:i + 1])

        df.alpha_r_inflow_int_pred[i + 1:t_next + 1], df.used_func_1[i + 1:t_next + 1] = func_dict[func](X1, w_prev,
                                                                                                         t_next - i,
                                                                                                         par)
        df.alpha_r_outflow_int_pred[i + 1:t_next + 1], df.used_func_2[i + 1:t_next + 1] = func_dict[func](X2, w_prev,
                                                                                                          t_next - i,
                                                                                                          par)

        flag_inflow = flag_outflow = 0
        for j in range(t_next - i):
            if np.isnan(df.r_inflow_int[i]):
                df.r_inflow_int_pred[i + 1 + j] = df.r_inflow_int_pred[i] - df.alpha_r_inflow_int_pred[t_next] * (j + 1)
                df.r_outflow_int_pred[i + 1 + j] = df.r_outflow_int_pred[i] - df.alpha_r_outflow_int_pred[t_next] * (
                            j + 1)
            else:
                df.r_inflow_int_pred[i + 1 + j] = df.r_inflow_int[i] - df.alpha_r_inflow_int_pred[t_next] * (j + 1)
                df.r_outflow_int_pred[i + 1 + j] = df.r_outflow_int[i] - df.alpha_r_outflow_int_pred[t_next] * (j + 1)
            if df.r_inflow_int_pred[i + 1 + j] < 0: flag_inflow = 1
            if df.r_outflow_int_pred[i + 1 + j] < 0: flag_outflow = 1

        if flag_inflow:
            mask = df.r_inflow_int[t_prev:i + 1].isna()
            X1 = df.r_inflow_int[t_prev:i + 1].where(~mask, df.r_inflow_int_pred[t_prev:i + 1])
            df.r_inflow_int_pred[i + 1:t_next + 1], _ = func_dict[func](X1, w_prev, t_next - i, par)
            df.used_func_1[i + 1] = 'power_r'

        if flag_outflow:
            mask = df.r_outflow_int[t_prev:i + 1].isna()
            X2 = df.r_outflow_int[t_prev:i + 1].where(~mask, df.r_outflow_int_pred[t_prev:i + 1])
            df.r_outflow_int_pred[i + 1:t_next + 1], _ = func_dict[func](X2, w_prev, t_next - i, par)
            df.used_func_2[i + 1] = 'power_r'

        if np.isnan(df.inflow_int[i]):
            df.inflow_int_pred[i + 1] = df.inflow_int_pred[i] * (df.r_inflow_int_pred[i + 1] + 1)
            df.outflow_int_pred[i + 1] = df.outflow_int_pred[i] * (df.r_outflow_int_pred[i + 1] + 1)
        else:
            df.inflow_int_pred[i + 1] = df.inflow_int[i] * (df.r_inflow_int_pred[i + 1] + 1)
            df.outflow_int_pred[i + 1] = df.outflow_int[i] * (df.r_outflow_int_pred[i + 1] + 1)

        for j in range(1, t_next - i):
            df.inflow_int_pred[i + 1 + j] = df.inflow_int_pred[i + j] * (df.r_inflow_int_pred[i + 1 + j] + 1)
            df.outflow_int_pred[i + 1 + j] = df.outflow_int_pred[i + j] * (df.r_outflow_int_pred[i + 1 + j] + 1)

        for j in range(t_next - i):
            df.Total_pred[i + 1 + j] = df.inflow_int_pred[i + 1 + j] - df.outflow_int_pred[i + 1 + j]
            df.error[i + 1 + j] = abs(df.Total_pred[i + 1 + j] - df.Total[i + 1 + j]) / df.Total[i + 1 + j] * 100
            df.error_C[i + 1 + j] = abs(df.inflow_int_pred[i + 1 + j] - df.inflow[i + 1 + j]) / df.inflow[
                i + 1 + j] * 100
            df.MAPE_all[i + 1 + j] = np.mean(df.error[i + 1:i + j + 2])
            df.MAPE_C_all[i + 1 + j] = np.mean(df.error_C[i + 1:i + j + 2])
        df.MAPE[t_next] = np.mean(df.error[i + 1:t_next + 1])
        df.MAPE_C[t_next] = np.mean(df.error_C[i + 1:t_next + 1])

    for i in range(t_start_ind, t_end_ind, w_next):
        df.mean_MAPE[i + w_next] = np.mean(df.error[i + 1:n_0])
    df.mean_MAPE[n_0 - 1] = df.MAPE[n_0 - 1]
    df['accuracy'] = 100 - df['error']
    df.accuracy[:n_0] = 100 - df.error[:n_0]
    return df


def errors_table(all_dfs, w, text, task, country, output_dir, t_start, t_end):
    if not all_dfs:
        return []

    df_sample = all_dfs[0]
    n_methods = df_sample.shape[1] - 2
    result_dfs = []

    for method_idx in range(n_methods):
        col_idx = method_idx + 2
        columns_from_all_dfs = [df.iloc[:, col_idx] for df in all_dfs]
        new_df = pd.concat(columns_from_all_dfs, axis=1)
        new_df.columns = [f'{i}' for i in w]
        new_df.insert(loc=0, column='Date', value=all_dfs[0]['Date'].values)
        new_df.insert(loc=0, column='Time', value=all_dfs[0]['Time'].values)
        result_dfs.append(new_df)

    output_path = os.path.join(output_dir, f'{country}_{task}_from({t_start}_to_{t_end}_{text}_predict.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        for i, df in enumerate(result_dfs, start=1):
            df.to_excel(writer, sheet_name=f'Method_{i}', index=False)
    return result_dfs


def linear_interpolate_duplicates(df):
    def process_column(column):
        result = column.copy()
        i = 0
        n = len(column)
        while i < n:
            current_value = column.iloc[i]
            j = i + 1
            while j < n and column.iloc[j] == current_value:
                j += 1
            if j - i > 1:
                start_value = current_value
                if j < n:
                    end_value = column.iloc[j]
                else:
                    end_value = column.iloc[i - 1] if i > 0 else start_value
                num_points = j - i
                if start_value != end_value:
                    step = (end_value - start_value) / num_points
                    for k in range(num_points):
                        result.iloc[i + k] = start_value + k * step
            i = j
        return result

    def not_smoothed_df(column):
        n = len(column)
        result = column.copy()
        for i in range(1, n):
            try:
                result.iloc[i] = column.iloc[i] / column.iloc[i - 1] - 1
            except:
                result.iloc[i] = 'inf'
        return result

    processed_df = df.copy()
    processed_df['inflow_not_smoothed'] = processed_df['inflow']
    processed_df['rate_inflow_not_smoothed'] = not_smoothed_df(processed_df['inflow_not_smoothed'])
    processed_df['outflow_not_smoothed'] = processed_df['outflow']
    processed_df['rate_outflow_not_smoothed'] = not_smoothed_df(processed_df['outflow_not_smoothed'])

    for col in ['inflow', 'outflow']:
        processed_df[col] = process_column(processed_df[col])

    while processed_df['inflow'][0] <= 0 or processed_df['outflow'][0] <= 0:
        processed_df = processed_df.drop(0).reset_index(drop=True)
    return processed_df


def data_C(df):
    col = ['Time', 'Date', 'C', 'R', 'I']
    df_country = df[col].reset_index()
    df_country = df_country.rename(columns={'C': 'inflow', 'R': 'outflow', 'I': 'Total'})
    return linear_interpolate_duplicates(df_country)


def create_output_dir(country, task, t_start, t_end, w_last, w_next, w_last_start):
    if w_last[0] == -1 and w_last[-1] == -1:
        output_dir = f'./{country}_{task}_prediction_from({t_start}_to_{t_end})_approx({w_last[0]}_to_{w_last[-1]})_with_aprx_start={w_last_start}_extrap={w_next}'
    else:
        output_dir = f'./{country}_{task}_prediction_from({t_start}_to_{t_end})_approx({w_last[0]}_to_{w_last[-1]})_extrap={w_next}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data(task, country):
    if task == 'C':
        return get_country_table(country, C, R_total)


task = 'C'
w_last_0 = 7
w_last_1 = 28
w_last = range(w_last_0, w_last_1 + 1)
w_last_start = 0
w_next = 14
t_start = 15
t_end = 1096

for country in selected_countries:
    try:
        data = load_data(task, country)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        sys.exit()

    for i in range(0, 14):
        output_dir = create_output_dir(country, task, t_start + i, t_end, w_last, w_next, w_last_start)

        all_error_dfs = []
        all_MAPE_dfs = []
        all_MAPE_all_dfs = []
        all_error_C_dfs = []
        all_MAPE_C_dfs = []
        all_MAPE_C_all_dfs = []

        for q in w_last:
            current_df_error, current_df_MAPE, current_df_MAPE_all, current_df_error_C, current_df_MAPE_C, current_df_MAPE_C_all = main(
                data, q, w_next, t_start + i, t_end, country, output_dir, w_last_start)
            all_error_dfs.append(current_df_error)
            all_MAPE_dfs.append(current_df_MAPE)
            all_MAPE_all_dfs.append(current_df_MAPE_all)
            all_error_C_dfs.append(current_df_error_C)
            all_MAPE_C_dfs.append(current_df_MAPE_C)
            all_MAPE_C_all_dfs.append(current_df_MAPE_C_all)

        errors_table(all_error_dfs, w_last, 'error', task, country, output_dir, t_start + i, t_end)
        errors_table(all_MAPE_dfs, w_last, 'MAPE', task, country, output_dir, t_start + i, t_end)
        errors_table(all_MAPE_all_dfs, w_last, 'MAPE_all', task, country, output_dir, t_start + i, t_end)
        errors_table(all_error_C_dfs, w_last, 'error_C', task, country, output_dir, t_start + i, t_end)
        errors_table(all_MAPE_C_dfs, w_last, 'MAPE_C', task, country, output_dir, t_start + i, t_end)
        errors_table(all_MAPE_C_all_dfs, w_last, 'MAPE_C_all', task, country, output_dir, t_start + i, t_end)

print(f"✅ Выполнение завершено!")