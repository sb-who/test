import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer


def winsorize(data, limit):
    mean = np.mean(data)
    std = np.std(data)
    data_post = data.copy()
    lower_bound = mean - std * limit
    upper_bound = mean + std * limit
    data_post[data_post < lower_bound] = lower_bound
    data_post[data_post > upper_bound] = upper_bound
    return data_post


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    data_post = (data - mean) / std
    return data_post


def factor_exposure(df_factor):
    if not isinstance(df_factor, pd.DataFrame):
        raise TypeError("输入数据必须为 pandas DataFrame 类型")

    # 必须包含的列
    required_columns = ['non_size', 'lncap', 'Hbeta', 'Hsigma_factor', 'RSTR', 'EGRO', 'SGRO', 'STOM', 'STOQ', 'STOA',
                        'DTOA', 'MLEV', 'BLEV', 'CTOP', 'ETOP', 'DASTD', 'CMRA', 'BTOP']

    # 检查 DataFrame 是否包含所有所需的列
    if not all(col in df_factor.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df_factor.columns]
        raise ValueError(f"DataFrame 缺少必要的列: {', '.join(missing_cols)}")

    # 使用 MICE 方法填充缺失值
    imputer = IterativeImputer(max_iter=10, random_state=0)
    df_filled = pd.DataFrame(imputer.fit_transform(df_factor), columns=df_factor.columns)

    # 创建一个新的 DataFrame 用于存储处理后的数据
    processed_data = {}
    for col in required_columns:
        if col in df_filled.columns:
            col_data = df_filled[col]
            col_w = winsorize(col_data, 3)
            col_n = normalize(col_w)
            processed_data[col] = col_n

    factor_r = pd.DataFrame(processed_data, columns=required_columns)
    return factor_r
def calculate_daily_returns_np(df):
    required_columns = ['stoke_id', 'price', 'prev_price', 'date']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 缺少必要的列: {', '.join(missing_cols)}")
    return True
    # 检查是否包含所有必要的列
    # 确保数据按 'stoke_id' 和 'date' 排序
    #df = df.sort_values(by=['stoke_id', 'date'])

    # 获取唯一的 stoke_id
    stoke_ids = df['stoke_id'].unique()
    # 计算不同 stoke_id 的数量
    num_stoke_ids = len(stoke_ids)
    # 初始化一个与原始数据结构相同的 DataFrame
    df['return'] = np.nan

    # 遍历每个 stoke_id
    for stoke_id in stoke_ids:
        # 提取当前 stoke_id 的数据
        stoke_data = df[df['stoke_id'] == stoke_id]

        # 转换为 numpy 数组
        prices = stoke_data['price'].values
        prev_prices = stoke_data['prev_price'].values

        # 计算收益率
        returns = np.empty(len(prices))
        returns[:] = np.nan
        # 使用价格和前一天的价格计算收益率
        returns[1:] = (prices[1:] - prev_prices[1:]) / prev_prices[1:]

        # 将计算结果添加到 DataFrame 中
        df.loc[df['stoke_id'] == stoke_id, 'return'] = returns

    return df, num_stoke_ids
#权重矩阵


def generate_daily_diagonal_matrices(df):
    # 检查是否存在 'MV' 列
    if 'MV' not in df.columns:
        raise ValueError("DataFrame 缺少 'MV' 列，无法生成对角矩阵。")

    # 获取日期列表并为每个日期计算总 MV
    dates = df['date'].values
    unique_dates = np.unique(dates)

    # 初始化结果列表
    matrices = []

    # 计算每日对角矩阵
    for date_idx, date in enumerate(unique_dates):
        df_date = df[df['date'] == date]
        num_stoke_ids = len(df_date['stoke_id'].unique())

        # 取 'MV' 列的平方根
        MV_sqrt = np.sqrt(df_date['MV'].values)

        total_MV = np.sum(MV_sqrt)
        MV_ratios = MV_sqrt / total_MV

        # 生成对角矩阵
        diagonal_matrix = np.diag(MV_ratios)

        matrices.append(diagonal_matrix)

    # 添加日期索引到原始 DataFrame
    df_with_matrices = df.copy()
    df_with_matrices['matrix_index'] = pd.factorize(df_with_matrices['date'])[0]

    return df_with_matrices, matrices
def industrial_factor(df):
    required_columns = ['b  ']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 缺少必要的列: {', '.join(missing_cols)}")
    return True
    # 检查是否包含所有必要的列
    # 确保数据按 'stoke_id' 和 'date' 排序
    #df = df.sort_values(by=['stoke_id', 'date'])

    # 获取唯一的 stoke_id
    industrial_ids = df['industrial_id'].unique()
    # 计算不同 stoke_id 的数量

    # 初始化一个与原始数据结构相同的 DataFrame
    df['industrial_MV'] = np.nan

    # 遍历每个 stoke_id
    for industrial_id in industrial_ids:
        # 提取当前 stoke_id 的数据
        stoke_data = df[df['industrial_id'] == industrial_id]

        # 转换为 numpy 数组
        MV = stoke_data['MV'].values
        prev_prices = stoke_data['prev_price'].values

        # 计算收益率
        returns = np.empty(len(prices))
        returns[:] = np.nan
        # 使用价格和前一天的价格计算收益率
        returns[1:] = (prices[1:] - prev_prices[1:]) / prev_prices[1:]

        # 将计算结果添加到 DataFrame 中
        df.loc[df['stoke_id'] == stoke_id, 'return'] = returns

    return df, num_stoke_ids
def calculate_mv_sum(df):
    # 确保 date 列为日期时间类型
    df['date'] = pd.to_datetime(df['date'])

    # 使用 groupby 和 sum 来计算每个日期和 industrial_id 组合的 MV 总和
    result = df.groupby(['date', 'industrial_id'])['MV'].sum().reset_index()

    return result
#约束矩阵
def create_diagonal_matrix_with_vector(df, n):


    df['date'] = pd.to_datetime(df['date'])
    daily_mv_sum = df.groupby(['date', 'industrial_id'])['MV'].sum().unstack(fill_value=0)
    total_mv_per_day = daily_mv_sum.sum(axis=1)
    ratio_matrix = daily_mv_sum.div(total_mv_per_day, axis=0)

    matrix_dict = {}

    for date in ratio_matrix.index:
        # 获取当天的比例数据
        daily_ratios = ratio_matrix.loc[date].values

        # 构建行向量，第一个元素前补 0
        row_vector = np.concatenate(([0], daily_ratios, [0] * (n - 28)))

        # 生成单位对角矩阵
        identity_matrix = np.eye(n)

        # 替换第二行
        identity_matrix[1] = row_vector

        # 将矩阵存入字典
        matrix_dict[date] = identity_matrix

    return matrix_dict
