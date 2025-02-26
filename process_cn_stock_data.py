import os
import pandas as pd
import numpy as np
import akshare as ak
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_indicators(data):
    """
    计算股票的技术指标
    
    参数:
        data: DataFrame, 包含OHLCV数据的DataFrame
    
    返回:
        DataFrame: 添加了技术指标的数据
    """
    # 确保日期列是索引
    if 'date' in data.columns:
        data = data.set_index('date')
    
    # 确保列名标准化
    data = data.rename(columns={
        '开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', 
        '成交量': 'Volume', '成交额': 'Amount'
    })
    
    # 添加日期特征
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    
    # 移动平均线
    data['MA5'] = data['Close'].shift(1).rolling(window=5).mean()
    data['MA10'] = data['Close'].shift(1).rolling(window=10).mean()
    data['MA20'] = data['Close'].shift(1).rolling(window=20).mean()
    
    # RSI指标
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD指标
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    # VWAP指标
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # 布林带
    period = 20
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['Std_dev'] = data['Close'].rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']
    
    # 相对大盘表现 (使用上证指数作为基准)
    try:
        benchmark_data = get_index_data('000001', data.index[0].strftime('%Y%m%d'), data.index[-1].strftime('%Y%m%d'))
        benchmark_data = benchmark_data.set_index('date')['Close']
        # 确保基准数据和股票数据有相同的日期
        common_dates = data.index.intersection(benchmark_data.index)
        if len(common_dates) > 0:
            data_aligned = data.loc[common_dates]
            benchmark_aligned = benchmark_data.loc[common_dates]
            data.loc[common_dates, 'Relative_Performance'] = (data_aligned['Close'] / benchmark_aligned) * 100
    except Exception as e:
        print(f"计算相对大盘表现时出错: {e}")
        data['Relative_Performance'] = np.nan
    
    # ROC指标
    data['ROC'] = data['Close'].pct_change(periods=1) * 100
    
    # ATR指标
    high_low_range = data['High'] - data['Low']
    high_close_range = abs(data['High'] - data['Close'].shift(1))
    low_close_range = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # 前一天数据
    data[['Close_yes', 'Open_yes', 'High_yes', 'Low_yes']] = data[['Close', 'Open', 'High', 'Low']].shift(1)
    
    # 删除缺失值
    data = data.dropna()
    
    return data

def get_stock_code(ticker):
    """
    将股票代码转换为AKShare格式
    
    参数:
        ticker: 股票代码或名称
    返回:
        标准化的股票代码
    """
    # 如果是纯数字，添加交易所前缀
    if ticker.isdigit():
        if ticker.startswith('6'):
            return f"sh{ticker}"
        elif ticker.startswith('0') or ticker.startswith('3'):
            return f"sz{ticker}"
        else:
            return ticker
    
    # 如果已经有前缀，标准化格式
    if ticker.startswith(('sh', 'sz', 'SH', 'SZ')):
        return ticker.lower()
    
    # 如果是股票名称，尝试查找对应代码
    try:
        stock_info = ak.stock_info_a_code_name()
        result = stock_info[stock_info['name'] == ticker]
        if not result.empty:
            code = result.iloc[0]['code']
            if code.startswith('6'):
                return f"sh{code}"
            else:
                return f"sz{code}"
    except:
        pass
    
    # 默认返回原始输入
    return ticker

def get_stock_data(ticker, start_date, end_date):
    """
    获取并处理单个A股股票的数据
    
    参数:
        ticker: 股票代码或名称
        start_date: 起始日期 (格式: YYYY-MM-DD)
        end_date: 结束日期 (格式: YYYY-MM-DD)
    返回:
        处理后的股票数据DataFrame
    """
    # 转换日期格式
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')
    
    # 获取标准化的股票代码
    stock_code = get_stock_code(ticker)
    
    # 下载股票数据
    try:
        # 使用AKShare获取A股历史数据
        if stock_code.startswith('sh'):
            exchange = 'SHSE'
            code = stock_code[2:]
        elif stock_code.startswith('sz'):
            exchange = 'SZSE'
            code = stock_code[2:]
        else:
            raise ValueError(f"不支持的股票代码格式: {stock_code}")
        
        # 使用AKShare的股票历史数据接口
        data = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                  start_date=start_date, end_date=end_date, 
                                  adjust="qfq")
        
        # 重命名列以匹配原始代码的期望
        data = data.rename(columns={
            '日期': 'date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume',
            '成交额': 'Amount'
        })
        
        # 确保日期列是日期时间格式
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        
        # 计算技术指标
        data = calculate_technical_indicators(data)
        
        return data
    except Exception as e:
        print(f"获取股票 {ticker} 数据时出错: {e}")
        return pd.DataFrame()

def get_index_data(index_code, start_date, end_date):
    """
    获取指数数据
    
    参数:
        index_code: 指数代码 (如: 000001 表示上证指数)
        start_date: 起始日期 (格式: YYYYMMDD)
        end_date: 结束日期 (格式: YYYYMMDD)
    返回:
        指数数据DataFrame
    """
    try:
        # 使用AKShare获取指数数据
        if index_code == '000001':  # 上证指数
            data = ak.stock_zh_index_daily(symbol="sh000001")
        elif index_code == '399001':  # 深证成指
            data = ak.stock_zh_index_daily(symbol="sz399001")
        elif index_code == '399300':  # 沪深300
            data = ak.stock_zh_index_daily(symbol="sz399300")
        else:
            raise ValueError(f"不支持的指数代码: {index_code}")
        
        # 筛选日期范围
        data['date'] = pd.to_datetime(data['date'])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        # 重命名列
        data = data.rename(columns={
            'open': 'Open',
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume'
        })
        
        return data
    except Exception as e:
        print(f"获取指数 {index_code} 数据时出错: {e}")
        return pd.DataFrame()

def clean_csv_files(file_path):
    """
    清理CSV文件，确保格式正确
    
    参数:
        file_path: CSV文件路径
    """
    try:
        df = pd.read_csv(file_path)
        
        # 确保日期列名为Date
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})
        
        # 保存修改后的文件
        df.to_csv(file_path, index=False)
        print(f"文件 {file_path} 已成功清理")
    except Exception as e:
        print(f"清理文件 {file_path} 时出错: {e}")
        raise

def main():
    """主函数：执行数据收集和处理流程"""
    # A股股票列表
    tickers = [
        '600519',  # 贵州茅台
        '601318',  # 中国平安
        '600036',  # 招商银行
        '000858',  # 五粮液
        '601166',  # 兴业银行
        '600276',  # 恒瑞医药
        '600887',  # 伊利股份
        '601888',  # 中国中免
        '600030',  # 中信证券
        '601398',  # 工商银行
        '000333',  # 美的集团
        '600000',  # 浦发银行
        '601288',  # 农业银行
        '600585',  # 海螺水泥
        '601668',  # 中国建筑
        '000651',  # 格力电器
        '600104',  # 上汽集团
        '601988',  # 中国银行
        '600028',  # 中国石化
        '601857',  # 中国石油
    ]

    # 设置参数
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    
    # 创建数据文件夹
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    
    # 获取并保存所有股票数据
    print("开始下载和处理A股数据...")
    successful_tickers = []
    for ticker in tickers:
        try:
            print(f"处理 {ticker} 中...")
            stock_data = get_stock_data(ticker, START_DATE, END_DATE)
            
            # 检查数据是否为空或者是否包含数据
            if stock_data.empty or len(stock_data) < 5:  # 至少需要5个交易日的数据
                print(f"{ticker} 没有足够的数据可供处理")
                continue
                
            # 保存数据
            output_path = f'{data_folder}/{ticker}.csv'
            stock_data.to_csv(output_path)
            
            try:
                clean_csv_files(output_path)
                successful_tickers.append(ticker)
                print(f"{ticker} 处理完成")
            except Exception as e:
                print(f"清洗 {ticker} 数据时出错: {str(e)}")
        except Exception as e:
            print(f"处理 {ticker} 时出错: {str(e)}")
    
    if successful_tickers:
        print(f"成功处理了以下股票: {', '.join(successful_tickers)}")
    else:
        print("没有成功处理任何股票数据，请检查网络连接")

if __name__ == "__main__":
    main() 