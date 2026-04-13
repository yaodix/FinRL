import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def test_calculate_gold_etf_features():
  df_30 = pd.read_csv('Rl_trading_demo/project_one/data/518880.SH.30m.csv')
  # set index
  df_30.set_index('trade_time', inplace=True)  # inplace=True, 直接修改原数据
  #去除amount列
  df_30.drop(columns=['amount'], inplace=True)


  print(df_30)

  #绘制
  df_30['close'][5500:].plot()
  plt.savefig('Rl_trading_demo/project_one/data_process/518880.SH.30m_close.png')

  # 检查是否有缺失值，null nan inf
  print('Number of Null Values =', df_30.isnull().sum())
  print('Number of na Values =', df_30.isna().sum())
  # 检查是否有inf值
  print('Number of inf Values =', df_30.isin([float('inf'), -float('inf')]).sum())
  print("="*50)
  print(f"{df_30.info()}")
  print(f"df describe {df_30.describe()}")   # 看最大最小值


def calculate_gold_etf_features(df):
    """
    计算黄金ETF第一、二层特征
    输入df需包含列: 'Open', 'High', 'Low', 'Close', 'Volume'
    输出: 包含所有特征的DataFrame
    """
    # 复制数据避免修改原始数据
    data = df.copy()
    
    # ========== 第一层：价格衍生特征 ==========
    
    # 收益率 (基础)
    data['Returns'] = data['Close'].pct_change()
    
    # 对数收益率 (用于波动率计算)
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 振幅 (衡量波动)
    data['Amplitude'] = (data['High'] - data['Low']) / data['Close'].shift(1)
    
    # 成交量比率 (相对20期均值)
    data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA20']
    data['Volume_Ratio'] = data['Volume_Ratio'].clip(0, 3)  # 限制异常值
    
    # 收盘位置 (K线实体在当日波动区间的位置)
    data['Close_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
    
    # ========== 第二层：技术指标 ==========
    
    # --- RSI (14周期) ---
    window = 14
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_Norm'] = data['RSI'] / 100  # 归一化到0-1
    
    # --- ATR (14周期) 和 ATR比率 ---
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    data['ATR_Ratio'] = data['ATR'] / data['Close']
    data['ATR_Ratio'] = data['ATR_Ratio'].clip(0, 0.1)  # 限制异常值
    
    # --- EMA差值 (MACD核心) ---
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['EMA_Diff'] = (data['EMA12'] - data['EMA26']) / data['Close']
    data['EMA_Diff'] = data['EMA_Diff'].clip(-0.1, 0.1)  # 限制异常值
    
    # --- 布林带宽度 (20周期, 2倍标准差) ---
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['STD20'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['MA20'] + (data['STD20'] * 2)
    data['BB_Lower'] = data['MA20'] - (data['STD20'] * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['MA20']
    data['BB_Width'] = data['BB_Width'].clip(0, 0.2)  # 限制异常值
    
    # --- 波动率 (20期年化波动率) ---
    data['Vol20'] = data['Log_Ret'].rolling(window=20).std() * np.sqrt(252 * 8)  # 30分钟数据，每天8个K线
    
    # 删除中间计算列（可选，保留一些常用列）
    columns_to_keep = [
        # 'Open', 'High', 'Low', 'Close', 'Volume',  # 原始数据
        'Open', "Close",  # 不训练
        'Returns', 'Amplitude', 'Volume_Ratio', 'Close_Position',  # 第一层
        'RSI_Norm', 'ATR_Ratio', 'EMA_Diff', 'BB_Width', 'Vol20'  # 第二层（归一化后）
    ]
    
    # 确保所有列都存在
    available_cols = [col for col in columns_to_keep if col in data.columns]
    data = data[available_cols]
    
    # 删除NaN值（因为滚动窗口会导致前几行为NaN）
    data = data.dropna()
    
    return data
  
def smart_feature_normalization(df, window=22*8, min_periods=20):
    """
    智能特征归一化：根据不同特征类型采用不同方法
    使用shift(1)完全避免未来信息泄露
    
    参数:
        df: 包含原始特征的DataFrame
        window: 滚动窗口（对于30分钟数据，8*22 ≈ 1个月）
        min_periods: 最小期数
    
    返回:
        包含归一化特征的DataFrame
    """
    df_result = df.copy()
    eps = 1e-8
    
    # ===== 第一类：需要Z-score的特征 =====
    zscore_features = ['Returns', 'Amplitude', 'ATR_Ratio', 'EMA_Diff', 'BB_Width', 'Vol20']
    
    for col in zscore_features:
        if col in df.columns:
            # 计算滚动均值和标准差
            rolling_mean = df[col].rolling(window=window, min_periods=min_periods).mean()
            rolling_std = df[col].rolling(window=window, min_periods=min_periods).std()
            
            # 关键：shift(1)确保只用过去数据
            mean_shifted = rolling_mean.shift(1)
            std_shifted = rolling_std.shift(1)
            
            # Z-score归一化
            df_result[f'{col}_Norm'] = (df[col] - mean_shifted) / (std_shifted + eps)
            df_result[f'{col}_Norm'] = df_result[f'{col}_Norm'].clip(-3, 3)
    
    # ===== 第二类：Volume_Ratio（用稳健统计量） =====
    if 'Volume_Ratio' in df.columns:
        # 计算滚动中位数和四分位数
        rolling_median = df['Volume_Ratio'].rolling(window=window, min_periods=min_periods).median()
        rolling_q75 = df['Volume_Ratio'].rolling(window=window, min_periods=min_periods).quantile(0.75)
        rolling_q25 = df['Volume_Ratio'].rolling(window=window, min_periods=min_periods).quantile(0.25)
        
        # shift(1)
        median_shifted = rolling_median.shift(1)
        q75_shifted = rolling_q75.shift(1)
        q25_shifted = rolling_q25.shift(1)
        
        # 计算IQR
        iqr_shifted = q75_shifted - q25_shifted
        
        # 稳健归一化
        df_result['Volume_Ratio_Norm'] = (df['Volume_Ratio'] - median_shifted) / (iqr_shifted + eps)
        df_result['Volume_Ratio_Norm'] = df_result['Volume_Ratio_Norm'].clip(-3, 3)
    
    # ===== 第三类：直接使用的特征（已归一化） =====
    if 'Close_Position' in df.columns:
        df_result['Close_Position_Norm'] = df['Close_Position']  # 保持原值
    
    # if 'RSI_Norm' in df.columns:
        # df_result['RSI_Norm_Final'] = df['RSI_Norm']  # 保持原值
        
    # 仅保留norm特征和open
    norm_features = [col for col in df_result.columns if col.endswith('_Norm') or col.endswith('_Norm_Final')]
    df_result = df_result[['Open', 'Close'] + norm_features]
    df_result = df_result.dropna()
    
    return df_result
  
def get_feat_split(src_dir, split_ratio=[0.75, 0.9], windows=22*8):
  '''
  src_dir: 数据文件路径,包含OHLCV列
  输出:
    df_train_norm, df_val_norm, df_test_norm: 归一化后的训练集和测试集，包含Open, Close, 归一化特征列
  '''
  df_30 = pd.read_csv(src_dir)
  df_30.set_index('trade_time', inplace=True)  # inplace=True, 直接修改原数据
  df_30.drop(columns=['amount'], inplace=True)
  df_30.columns = df_30.columns.str.title()
  
  df_30_features = calculate_gold_etf_features(df_30)
  
  split_size_train = int(len(df_30_features) * split_ratio[0])
  split_size_test = int(len(df_30_features) * split_ratio[1])
  df_feat_head = df_30_features[:split_size_train]
  df_feat_val = df_30_features[split_size_train:split_size_test]
  df_feat_tail = df_30_features[split_size_test:]
  
  df_feat_head_norm = smart_feature_normalization(df_feat_head, windows)
  df_feat_val_norm = smart_feature_normalization(df_feat_val, windows)
  df_feat_tail_norm = smart_feature_normalization(df_feat_tail, windows)
  
  print(f"train_size: {len(df_feat_head_norm)}")
  print(f"val_size: {len(df_feat_val_norm)}")
  print(f"test_size: {len(df_feat_tail_norm)}")
  
  return df_feat_head_norm, df_feat_val_norm, df_feat_tail_norm


if __name__ == '__main__':
  df_30_dir = 'Rl_trading_demo/project_one/data/518880.SH.30m.csv'
  df_train_norm, df_val_norm, df_test_norm = get_feat_split(df_30_dir, split_ratio= [0.75, 0.9])
  print(df_train_norm.info())
  print(df_test_norm.info())
