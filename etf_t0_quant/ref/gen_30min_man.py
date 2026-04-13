import pandas as pd
import numpy as np

df = pd.read_csv("Rl_trading_demo/project_one/data/518880.SH.1m.csv")
df.columns.values[0] = 'trade_time'

# volume倒数31到61的索引求和
# print(df.iloc[-241:-210])
# print(df['volume'].iloc[-241:-210].sum())
# exit(0)

# 1. 预处理时间列
df['trade_time'] = pd.to_datetime(df['trade_time'], format='%Y%m%d%H%M%S')
df.set_index('trade_time', inplace=True)

# 2. 定义聚合规则
agg_rules = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'amount': 'sum'
}

# 3. 创建时间分组标签函数
def get_30min_label(timestamp):
    """根据交易时间返回对应的30分钟K线标签"""
    time_str = timestamp.strftime('%H:%M')
    
    # 上午
    if '09:30' <= time_str <= '10:00':
        return timestamp.replace(hour=10, minute=0, second=0, microsecond=0)
    elif '10:00' < time_str <= '10:30':
        return timestamp.replace(hour=10, minute=30, second=0, microsecond=0)
    elif '10:30' < time_str <= '11:00':
        return timestamp.replace(hour=11, minute=0, second=0, microsecond=0)
    elif '11:00' < time_str <= '11:30':
        return timestamp.replace(hour=11, minute=30, second=0, microsecond=0)
    
    # 下午
    elif '13:00' < time_str <= '13:30':
        return timestamp.replace(hour=13, minute=30, second=0, microsecond=0)
    elif '13:30' < time_str <= '14:00':
        return timestamp.replace(hour=14, minute=0, second=0, microsecond=0)
    elif '14:00' < time_str <= '14:30':
        return timestamp.replace(hour=14, minute=30, second=0, microsecond=0)
    elif '14:30' < time_str <= '15:00':
        return timestamp.replace(hour=15, minute=0, second=0, microsecond=0)
    
    return None

# 4. 过滤交易时间
df_trade = df[(df.index.time >= pd.Timestamp('09:30:00').time()) & 
              (df.index.time <= pd.Timestamp('15:00:00').time())].copy()

# 5. 添加分组标签
df_trade['group_label'] = df_trade.index.map(get_30min_label)

# 6. 删除无法分组的数据
df_trade = df_trade.dropna(subset=['group_label'])

# 7. 按标签聚合
df_30m = df_trade.groupby('group_label').agg(agg_rules)
df_30m.index.name = 'trade_time'

# 8. 只删除完全没有数据的行（保留有0值的行）
df_30m = df_30m.dropna(how='all')
df_30m.to_csv("Rl_trading_demo/project_one/data/518880.SH.30m.csv")

# 9. 检查结
# print(df_30m.tail(20))

# 10 对比
df_1000 = pd.read_csv("Rl_trading_demo/project_one/data/518880.SH.30m_10000.csv")
len_df = len(df_1000['volume'])-1  # 索引0会不一致为什么？ 

# 对比所有列
columns_to_compare = ['open', 'high', 'low', 'close', 'volume', 'amount']
for col in columns_to_compare:
    s1 = df_30m[col].tail(len_df).reset_index(drop=True)
    s2 = df_1000[col].tail(len_df).reset_index(drop=True)
    # 使用绝对差值判断是否有显著差异（大于0.00001）
    diff_res = ((s1 - s2).abs() > 0.00001).any()
    # 把不一样的行打印出来
    diff_rows = (s1 - s2).abs() > 0.00001
    print(f'\n=== {col} {len(s1)}个对比结果有{diff_rows.sum()}个不一致: {"不一致" if diff_res else "一致"} ===')
    if diff_rows.any():
        # 同时显示两个数据源的值
        diff_df = pd.DataFrame({
            'df_30m': s1[diff_rows],
            'df_1000': s2[diff_rows]
        })
        print(diff_df)
    else:
        print('所有值一致')
