import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

class Halflife:
    def __init__(self, half_life):
        self.half_life = half_life

    def compute_alpha(self):
        return 1 - np.exp(np.log(0.5) / self.half_life)

    def compute_ewma_with_weights(self, data, window_size):
        alpha = self.compute_alpha()  # 计算平滑因子
        n = data.shape[0]
        m = data.shape[1]

        ewma_values = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
        weights_matrix = pd.DataFrame(np.nan, index=data.index, columns=[f'Weight_{i}' for i in range(window_size)])

        for col in data.columns:
            column_data = data[col].values
            ewma = np.zeros(n)
            weights_full_matrix = np.zeros((n, window_size))

            for i in range(n):
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
                data_window = column_data[start_idx:end_idx]

                if len(data_window) < window_size:
                    # 对于不足的窗口长度，计算滑动平均
                    average = np.mean(data_window)
                    weights = np.ones(len(data_window))  # 使用均等权重
                    weights_padded = np.zeros(window_size)
                    weights_padded[window_size - len(weights):] = weights

                    # 计算EWMA（滑动平均）
                    ewma[i] = average

                    # 更新权重矩阵
                    weights_full_matrix[i] = weights_padded
                else:
                    # 对于足够的窗口长度，计算EWMA
                    weights = np.array([(1 - alpha) ** (window_size - j - 1) for j in range(window_size)])
                    weights = weights[::-1]

                    ewma[i] = np.sum(data_window * weights[-len(data_window):]) / np.sum(weights[-len(data_window):])

                    weights_full_matrix[i] = weights

            ewma_values[col] = ewma
            weights_matrix[col] = list(weights_full_matrix)

        return ewma_values, weights_matrix


def winsorize(data, limit):
    mean = np.mean(data)
    std = np.std(data)
    data_post = data.copy()
    data_post = data[data < mean-std*limit] = mean-std*limit
    data_post = data[data > mean + std * limit] = mean + std * limit
    return data_post


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    data_post = (data-mean)/std
    return data_post


def styles(data):
    result['Beta'] = data.BETA
    result['Momentum'] = data.RSTR
    result['Size'] = data.LNCAP
    result['RV'] = 0.74 * data.DASTD + 0.16 * data.CMRA + 0.1 * data.HSIGMA
    result['NLS'] = data.NLSIZE
    result['BTP'] = data.BTOP
    result['Liquidity'] = 0.35 * data.STOM + 0.35 * data.STOQ + 0.3 * data.STOA
    result['EY'] = 0.68 * data.EPFWD + 0.21 * data.CETOP + 0.11 * data.ETOP
    result['Growth'] = 0.18 * data.EGRLF + 0.11 * data.EGRSF
    result['Leverage'] = 0.38 * data.MLEV + 0.35 * data.DTOA + 0.27 * data.BLEV
    return result


def fill_data(data, column, limit):
    data = data[np.isfinite(data.iloc[:, column])]
    mean = np.mean(data.iloc[:, column])
    std = np.std(data.iloc[:, column])
    idx = (data.iloc[:, column] > (mean - std * limit)) & (data.iloc[:, column] < (mean + std * limit))
    data_new = data[idx]
    return data_new


def linenar_regression(x, y):
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    # 计算协方差和方差
    cov_xy = np.mean((X - x_mean) * (y - y_mean))
    var_x = np.mean((X - x_mean) ** 2)
    # 计算斜率和截距
    coef = cov_xy / var_x
    intercept = y_mean - beta_1 * x_mean
    return coef, intercept


def cubed(data):
    cubed_data = data**3
    slope, intercept = linenar_regression(data, cubed_data)
    factor_1 = cubed_data-(slope*data+intercapt)
    factor_2 = fill_data(factor_1, limit)
    factor_3 = winsorize(factor_2, limit)
    factor = normalize(factor_3)
    return factor


def neutral(data,industry,factor):
    industry_factor = []
    for i in industry:
        industry_factor.append[i]   #筛选条件
    xvars = data['marketcap'] + industry_factor
    yvars = df[xvars+[factor]]


    if yvars.isnull().values.any():
        warnings.warn("missing_value")
        used_factors_df = yvars.dropna()
    slope,res  = linenar_regression(xvars,yvars)
    return res


def shift(data,t):
    date_shift = data.shift(t)
    return date_shift


class Date_transfor():

    def __init__(self):
        pass

    def get_last_entry(self, df):
        """
        获取最后一行数据。
        """
        return df.iloc[-1]

    def transfer_to_annual(self, data):
        """
        将数据从较高频率转换为年度频率，并选择每年最后一个数据点。
        """
        grouped_data = data.groupby(pd.Grouper(freq='y'))
        data_annual = grouped_data.apply(self.get_last_entry)  #将最后一个季度的数据作为当年的数据
        return data_annual

    def shift(self, data, t):
        date_shift = data.shift(t)
        return date_shift

    def fill_annual_to_daily(self, annual_data, daily_data):
        """
        将上一年年末的数据填充到新一年。
        使用日期索引处理数据。
        """
        filled_data = daily_data.copy()

        for year in range(annual_data.index.year[0] + 1, annual_data.index.year[-1] + 1):
            start_of_year = pd.Timestamp(year=year, month=1, day=1)
            end_of_previous_year = pd.Timestamp(year=year - 1, month=12, day=31)

            if end_of_previous_year in annual_data.index:
                value_to_fill = annual_data.loc[end_of_previous_year].values[0]
                print(f"Filling year {year} with value {value_to_fill} from previous year")
                mask = (daily_data.index >= start_of_year) & (daily_data.index < start_of_year + pd.DateOffset(years=1))
                filled_data.loc[mask, 'value'] = value_to_fill

        return filled_data

    def fill_first_year_with_q1_data(self, quarterly_data, daily_data):
        """
        将第一年第一季度的数据填充到第一年的缺失值。
        使用顺序索引处理数据。
        """
        quarterly_data_seq = quarterly_data.reset_index(drop=True)
        first_year = quarterly_data.index[0].year
        first_quarter_value = None

        # 查找第一个非 NaN 的季度数据
        for idx in range(len(quarterly_data_seq)):
            value = quarterly_data_seq.iloc[idx]
            if not pd.isna(value).all():
                if idx > 3:  # 如果第一个非 NaN 数据的索引大于 3
                    print("跳过填充，因为第一个有效的季度数据索引大于 3")
                    return daily_data
                first_quarter_value = value
                break

        if first_quarter_value is None:
            raise ValueError("无法找到有效的季度数据进行填充。")

        first_year_mask = (daily_data.index >= pd.Timestamp(year=first_year, month=1, day=1)) & (
                daily_data.index < pd.Timestamp(year=first_year + 1, month=1, day=1))
        daily_data.loc[first_year_mask, 'value'] = first_quarter_value.values[0]

        return daily_data
    def fill_quarterly_to_daily(self, quarterly_data, daily_data):
        """
        将季度频数据填充到每日频数据。
        参数:
        quarterly_data (pandas.Series): 季度频率的数据，具有日期索引
        daily_data (pandas.DataFrame): 每日频率的数据，具有日期索引

        返回:
        pandas.DataFrame: 填充后的每日数据
        """
        if not isinstance(quarterly_data.index, pd.DatetimeIndex) or not isinstance(daily_data.index, pd.DatetimeIndex):
            raise TypeError("输入数据的索引必须为 DateTimeIndex 类型")

        filled_data = daily_data.copy()

        for quarter_end in quarterly_data.index:
            start_of_quarter = quarter_end - pd.DateOffset(months=2)  # 假设季度末是每季度的最后一天
            start_of_quarter = start_of_quarter.replace(day=1)
            end_of_quarter = quarter_end

            if end_of_quarter in quarterly_data.index:
                value_to_fill = quarterly_data.loc[end_of_quarter]
                mask = (daily_data.index >= start_of_quarter) & (daily_data.index <= end_of_quarter)
                filled_data.loc[mask, 'value'] = value_to_fill

        return filled_data

    def summarize_by_fourth_quarter(df, value_columns):
        """
        transfer_to_annual
        根据第四季度汇总指定列的数据。

        参数:
        df (pd.DataFrame): 包含 `timestamp` 列和多个 `value` 列的 DataFrame。
        value_columns (list of str): 要汇总的 `value` 列名列表。

        返回:
        pd.DataFrame: 包含年度汇总值的 DataFrame。
        """
        # 提取年份和季度信息
        df['year'] = df['timestamp'].dt.year
        df['quarter'] = df['timestamp'].dt.quarter

        # 获取唯一的年份
        unique_years = np.unique(df['year'].to_numpy())

        # 初始化汇总结果
        annual_summaries = {col: np.zeros(unique_years.shape[0]) for col in value_columns}

        # 对每个唯一年份进行处理
        for i, year in enumerate(unique_years):
            # 找到该年份的第四季度数据
            mask = (df['year'] == year) & (df['quarter'] == 4)

            # 汇总该年份第四季度的指定列数据
            for col in value_columns:
                annual_summaries[col][i] = np.sum(df.loc[mask, col])

        # 创建年度汇总 DataFrame
        annual_df = pd.DataFrame({
            'year': unique_years
        })

        for col in value_columns:
            annual_df[f'total_{col}'] = annual_summaries[col]

        return annual_df


def fill_quarterly_to_daily(df_quarterly, df_daily, value_columns):
    """
    将季度频数据填补为日频数据的时间点，其中每个季度的数据根据季度的时间范围填充到日频数据中。

    参数:
    df_quarterly (pd.DataFrame): 包含季度频率数据的 DataFrame，包括时间戳列 'timestamp' 和数值列
    df_daily (pd.DataFrame): 包含日频数据的 DataFrame，包括时间戳列 'date' 和数值列
    value_columns (list of str): 需要填补的数值列名列表

    返回:
    pd.DataFrame: 用季度数据填充的日频数据 DataFrame
    """

    # 确保数据按时间排序
    df_quarterly = df_quarterly.sort_values('timestamp')
    df_daily = df_daily.sort_values('date')

    # 将季度数据转换为 numpy 数组
    quarter_dates = df_quarterly['timestamp'].to_numpy()

    # 创建一个用于存储日频数据的字典
    daily_filled = {col: np.full(len(df_daily), np.nan) for col in value_columns}

    for col in value_columns:
        # 遍历每个季度数据
        for i in range(len(quarter_dates)):
            start_date = quarter_dates[i]

            # 找到当前季度的结束日期
            if i < len(quarter_dates) - 1:
                end_date = quarter_dates[i + 1] - pd.Timedelta(days=1)
            else:
                # 确保生成的季度数据覆盖完整的日频日期，包括最后一天
                end_date = pd.date_range(start=start_date, periods=1, freq='Q').max()

            # 找到日频数据中在当前季度范围内的索引
            mask = (df_daily['date'] >= start_date) & (df_daily['date'] <= end_date)
            daily_filled[col][mask] = df_quarterly[col].iloc[i]

    # 将填充的结果加入到日频数据 DataFrame
    for col in value_columns:
        df_daily[col] = daily_filled[col]

    return df_daily


class Return:
    def __init__(self):
        pass

    def daily_return(self, prev_close, close):
        """
        计算每日收益率
        """
        return (close - prev_close) / prev_close

    def compute_daily_returns(self, data):
        """
        计算每日收益率，并返回包含每日收益率的 DataFrame
        假设 data 包含 'Prev_Close' 和 'Close' 列
        """
        # 确保 DataFrame 中有 'Prev_Close' 和 'Close' 列
        if 'Prev_Close' not in data.columns or 'Close' not in data.columns:
            raise ValueError("DataFrame must contain 'Prev_Close' and 'Close' columns.")

        # 计算每日收益率
        data['Daily_Return'] = data.apply(lambda row: self.daily_return(row['Prev_Close'], row['Close']), axis=1)

        # 返回包含每日收益率的 DataFrame
        daily_returns_df = data[['Daily_Return']]
        return daily_returns_df

    def monthly_return(self, data):
        """
        计算月度收益率
        """
        daily_returns_df = self.compute_daily_returns(data)  # 获取每日收益率的 DataFrame
        mon_returns = daily_returns_df['Daily_Return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        return mon_returns

    def year_return(self, data):
        """
        计算年度收益率
        """
        daily_returns_df = self.compute_daily_returns(data)  # 获取每日收益率的 DataFrame
        # 计算月度收益率
        monthly_returns_df = daily_returns_df.resample('M').apply(lambda x: (1 + x).prod() - 1)
        # 计算年度收益率
        year_returns = monthly_returns_df.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        return year_returns
def cumulative_by_year(data, value_col):
    """
    计算按月度的累计值，达到12个月后重新累计
    :param data: 输入的 DataFrame，要求包含月度数据
    :param value_col: DataFrame 中需要计算累计值的列名
    :return: 按月累计的结果 DataFrame
    """
    # 确保数据按月度频率
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame with monthly frequency.")

    if value_col not in data.columns:
        raise ValueError(f"Column '{value_col}' not found in the DataFrame.")

    # 确保索引是日期类型
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)

    # 计算累计值
    data['Cumulative'] = data[value_col].cumsum()
    #可以按照日频数据处理
    #monthly_cumulative = data['Cumulative'].resample('M').last()
    # 计算每月累计值
    monthly_cumulative = data['Cumulative']

    # 初始化累计值和月份计数器
    cumulative = []
    current_cumulative = 0
    month_counter = 0

    for value in monthly_cumulative:
        month_counter += 1
        current_cumulative = value
        if month_counter % 12 == 0:
            cumulative.append(current_cumulative)
            current_cumulative = 0
        else:
            cumulative.append(current_cumulative)

    # 如果数据点的数量不是12的倍数，追加最后的累计值
    if month_counter % 12 != 0:
        cumulative.append(current_cumulative)

    # 创建结果 DataFrame
    # 索引调整为每12个月的最后一天
    result_index = monthly_cumulative.index[:len(cumulative)]
    result = pd.DataFrame({'Cumulative': cumulative}, index=result_index)

    return result



