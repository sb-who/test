import pandas as pd
import numpy as np
import basement     as bs
class Size():
    def __init__(self):
        self.TotalMV = df['NeoMV']

    def size_factor(self):

        lncap  = np.log(self.TotalMV+0.000001)
        return lncap
    def non_lin_size(self):
        lncap = self.size_factor()
        non_size = cubed(lncap)
        return non_size

class Beta():
    def __init__(self,r,rm):
        self.r = r
        self.rm = rm
        self.Hbeta = None
        self.Hsigma = None
    def Hbeta_regression(self):
        Hbeta, Hsigma = bs.linenar_regression(self.r,self.rm)
        return self.Hbeta, self.Hsigma
    def Hsigma(self):
        if self.Hsigma is None:
            raise ValueError("Hsigma is not computed")
        Hsigma_factor = np.std(Hsigma)
        return Hsigma_factor
class Momentum():
    def __init__(self, p_price, price, rm, half_life):
        """
                    Parameters:
                    df (pandas.DataFrame): 包含TTlib、TTasse ld PE BE ME

                    Raises:
                    TypeError: 如果输入参数不是 DataFrame，则抛出异常
                    ValueError: 如果 DataFrame 不包含所需的列，则抛出异常
                """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入数据必须为 pandas DataFrame 类型")

        # 必须包含的列
        required_columns = ['p_price', 'price', 'rm']

        # 检查 DataFrame 是否包含所有所需的列
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"DataFrame 缺少必要的列: {', '.join(missing_cols)}")
        self.df = df
        self.Halflife = bs.Halflife
        self.half_life = half_life
        self.Return = bs.Return
        self.rm = rm
    def RSTR(self,window_size,n_exclude):
        r_daily = self.Return.compute_daily_returns(df)
        rdf = self.rm
        excess_return = log(1+r_daily) - log(1+rdf)
        RSTR1 = self.Halflife.compute_ewma_with_weights(excess_return,window_size=504)
        RSTR = []
        for i in range(window_size - n_exclude - 1):
            RSTR.append(0)
        for i in range(len(RSTR1) - window_size + 1):
            # 当前窗口
            window = nums[i:i + window_size]
            # 去掉最近的n_exclude个数
            window_excluded = window[:-n_exclude]
            # 计算窗口中剩余数的和
            total_sum = sum(window_excluded)
            RSTR.append(total_sum)
        return RSTR


class Gross():
    def __init__(self,eps,revenue):
        self.eps = eps
        self.revenue = revenue
    def EGRO(self):
        yvar = self.eps.copy()
        EGRO = []
        for i in range(19,len(self.eps)):
            y = yvar.iloc[i-19,i+1] #将年度数据变为季度数据 5年对应20个季度
            x = np.arange(20)
            slope,_ = bs.linear_regression_coef(x, y)
            mean = y.mean()
            if mean == 0:
                EGRO_values = slope
            else:
                EGRO_values = slope / mean
            EGRO.append(EGRO_values)
        return EGRO
    def SGRO(self):
        yvar = self.eps.copy()
        SGRO = []
        for i in range(19,len(self.revenue)):
            y = yvar.iloc[i-19,i+1] #将年度数据变为季度数据 5年对应20个季度
            x = np.arange(20)
            slope,_ = bs.linear_regression_coef(x, y)
            mean = y.mean()
            if mean == 0:
                SGRO_values = slope
            else:
                SGRO_values = slope / mean
            SGRO.append(SGRO_values)



class Liquidity():
    def __init__(self, df):
        """
        Parameters:
        df (pandas.DataFrame): 包含 FFTTM、FFTRMThree 和 FFTRY 列的数据框

        Raises:
        TypeError: 如果输入参数不是 DataFrame，则抛出异常
        ValueError: 如果 DataFrame 不包含所需的列，则抛出异常
        """
        # 检查是否为 DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入数据必须为 pandas DataFrame 类型")

        # 必须包含的列
        required_columns = ['FFTTM', 'FFTRMThree', 'FFTRY']

        # 检查 DataFrame 是否包含所有所需的列
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"DataFrame 缺少必要的列: {', '.join(missing_cols)}")

        self.df = df
        self.epsilon = 0.00001  # 避免log(0)的问题
    def STOM(self):
        STOM = np.log(self.df['FFTTM']+self.epsilon)
        return STOM
    def STOQ(self):
        STOQ = np.log(self.df['FFTRMTHree']+self.epsilon)
        return STOQ
    def STOA(self):
        STOA = np.log(self.df['FFTRY']+self.epsilon)
        return STOA
class Leverage():
    def __init__(self, df):
        """
            Parameters:
            df (pandas.DataFrame): 包含TTlib、TTasse ld PE BE ME

            Raises:
            TypeError: 如果输入参数不是 DataFrame，则抛出异常
            ValueError: 如果 DataFrame 不包含所需的列，则抛出异常
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入数据必须为 pandas DataFrame 类型")

        # 必须包含的列
        required_columns = ['TTlib', 'TTasse', 'ld', 'PE', 'BE', 'ME']

        # 检查 DataFrame 是否包含所有所需的列
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"DataFrame 缺少必要的列: {', '.join(missing_cols)}")
        self.df = df
        self.epsilon = 0.00001
        self.data_transformer = bs.Data_transfor()  # 实例化 Data_transfor 类
    def DTOA(self):
        totaldebt1 = self.data_transfor.transfer_to_annual(df['TTlib'])
        totaldebt = self.data_transfor,shift(totaldebt1,1)
        totalassets1 = self.data_transfor.transfer_to_annual(df['TTasse'])
        totalassets = self.data_transfor, shift(totalassets1, 1)
        DTOA = totaldebt / (totalassets+self.epsilon)
        return DTOA
    def BLEV(self):
        ld1 = self.data_transfor.transfer_to_annual(df['ld'])
        ld = self.data_transfor, shift(totaldebt1, 1)
        PE1 = self.data_transfor.transfer_to_annual(df['PE'])
        PE = self.data_transfor, shift(totalassets1, 1)
        BE1 = self.data_transfor.transfer_to_annual(df['BE'])
        BE = self.data_transfor, shift(totalassets1, 1)
        BLEV = (ld+PE+BE)/(BE+self.epsilon)
        return BLEV
    def MLEV(self):
        ld1 = self.data_transfor.transfer_to_annual(df['ld'])
        ld = self.data_transfor, shift(totaldebt1, 1)
        PE1 = self.data_transfor.transfer_to_annual(df['PE'])
        PE = self.data_transfor, shift(totalassets1, 1)
        ME1 = self.data_transfor.transfer_to_annual(df['ME'])
        ME = self.data_transfor, shift(totalassets1, 1)
        BLEV = (ld + PE + ME) / (ME + self.epsilon)
        return MLEV
class earnings_yield():
    def __init__(self, df):
        """
            Parameters:
            df (pandas.DataFrame): 包含TTlib、TTasse ld PE BE ME

            Raises:
            TypeError: 如果输入参数不是 DataFrame，则抛出异常
            ValueError: 如果 DataFrame 不包含所需的列，则抛出异常
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入数据必须为 pandas DataFrame 类型")

        # 必须包含的列
        required_columns = ['NOCF', 'Fixexp', 'NegotiableMV', 'Netprofit','date']

        # 检查 DataFrame 是否包含所有所需的列
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"DataFrame 缺少必要的列: {', '.join(missing_cols)}")
        self.df = df
        self.data_transformer = bs.Data_transfor()
        self.epsilon = 0.00001
    def CETOP(self):
        # 确保日期列是 datetime 类型
        df['date'] = pd.to_datetime(df['date'])
        # 将日期列设置为索引
        df.set_index('date', inplace=True)
        #转变数据频次
        daily_data_NegotiableMV = df[['NegotiableMV']] #日频数据
        quarterly_data_NOCF = df[['NOCF']].resample('Q').last() #季度频
        quarterly_data_Fixexp = df[['Fixexp']].resample('Q').last()  # 季度频
        #NOCF处理
        annual_data_NOCF = self.date_transfor.transfer_to_annual(quarterly_data_NOCF)  #季度频转为年度频
        filled_data_NOCF = self.date_transfor.fill_annual_to_daily(annual_data_NOCF, daily_data_NegotiableMV)# 将年度频数据按日填充 向后填充
        final_filled_data_NOCF = self.date_transfor.fill_first_year_with_q1_data(quarterly_data_NOCF, filled_data_NOCF) #第一年缺失值填充
        #Fixexp处理
        annual_data_Fixexp = self.date_transfor.transfer_to_annual(quarterly_data_Fixexp)  # 季度频转为年度频
        filled_data_Fixexp = self.date_transfor.fill_annual_to_daily(annual_data_Fixexp, daily_data_NegotiableMV)  # 将年度频数据按日填充 向后填充
        final_filled_data_Fixexp = self.date_transfor.fill_first_year_with_q1_data(quarterly_data_Fixexp,
                                                                            filled_data_Fixexp)  # 第一年缺失值填充
        CETOP = (final_filled_data_NOCF - final_filled_data_Fixexp)/(self.epsilon+daily_data_NegotiableMV)
        return CETOP
    def ETOP(self):
        # 确保日期列是 datetime 类型
        df['date'] = pd.to_datetime(df['date'])
        # 将日期列设置为索引
        df.set_index('date', inplace=True)
        # 转变数据频次
        daily_data_NegotiableMV = df[['NegotiableMV']]  # 日频数据
        quarterly_data_Netprofit = df[['Netprofit']].resample('Q').last()  # 季度频
        # NOCF处理
        annual_data_Netprofit = self.date_transfor.transfer_to_annual(quarterly_data_Netprofit)  # 季度频转为年度频
        filled_data_Netprofit = self.date_transfor.fill_annual_to_daily(annual_data_Netprofit,
                                                                   daily_data_NegotiableMV)  # 将年度频数据按日填充 向后填充
        final_filled_data_Netprofit = self.date_transfor.fill_first_year_with_q1_data(quarterly_data_Netprofit,
                                                                                 filled_data_Netprofit)  # 第一年缺失值填充
        ETOP = final_filled_data_Netprofit/(self.epsilon+daily_data_NegotiableMV)
        return ETOP
class Residual_volatility():
    def __init__(self, p_price, price, rm, half_life):
        """
                    Parameters:
                    df (pandas.DataFrame): 包含TTlib、TTasse ld PE BE ME

                    Raises:
                    TypeError: 如果输入参数不是 DataFrame，则抛出异常
                    ValueError: 如果 DataFrame 不包含所需的列，则抛出异常
                """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入数据必须为 pandas DataFrame 类型")

        # 必须包含的列
        required_columns = ['p_price', 'price', 'rm']

        # 检查 DataFrame 是否包含所有所需的列
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"DataFrame 缺少必要的列: {', '.join(missing_cols)}")
        self.df = df
        self.Halflife = bs.Halflife
        self.half_life = half_life
        self.Return = bs.Return
        self.rm = rm
    def DASTD(self,window_size):
        r_daily = self.Return.compute_daily_returns(df)
        excess = r-self.rm
        DTASD1 = self.Halflife.compute_ewma_with_weights(excess,window_size)
        DTASD = np.std(DTASD1)
        return DTASD
    def CMRA(self):
        r_monthly = self.Return.monthly_return(df)
        rm_monthly = df['rm'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        CMRA2 = np.log(r_monthly+1)-np.log(1+rm_monthly)
        CMRA1 = bs.cumulative_by_year(CMRA2)
        yearly_summary = cumulative_df.resample('Y').agg(['max', 'min'])
        yearly_summary.columns = ['Year_Max', 'Year_Min']
        CMRA = yearly_summary['year_Max'] -  yearly_summary['year_Min']
        return CMRA
class BTOP(self):
    def __init__(self, commonSE, NegotiableMV):
        """
                    Parameters:
                    df (pandas.DataFrame): 包含commonSE, NegotiableMV

                    Raises:
                    TypeError: 如果输入参数不是 DataFrame，则抛出异常
                    ValueError: 如果 DataFrame 不包含所需的列，则抛出异常
                """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入数据必须为 pandas DataFrame 类型")

        # 必须包含的列
        required_columns = ['p_price', 'price', 'rm']

        # 检查 DataFrame 是否包含所有所需的列
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"DataFrame 缺少必要的列: {', '.join(missing_cols)}")
        self.commonSE = commonSE
        self.NegotiableMV = NegotiableMV
        self.Date_transfor = bs.Date_transfor
        self.epsilon = 0.00001
    def BTOP(self):
        commonSE1 = self.Date_transfor.fill_quarterly_to_daily(self.commonSE,self.NegotiableMV)
        BTOP = commonSE1/(self.epsilon+self.NegotiableMV)
        return BTOP

