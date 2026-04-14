# 30mink线与同花顺30分钟K线数据对比一致
import os
from dotenv import load_dotenv
from tickflow import TickFlow
import datetime


load_dotenv()
tickflow_apikey = os.getenv("tickflow_apikey")


tf = TickFlow(api_key=tickflow_apikey)

# 将日期转换为毫秒时间戳
start = int(datetime.datetime(2024, 1, 1).timestamp() * 1000)
end = int(datetime.datetime(2026, 4, 14).timestamp() * 1000)

doupo_etf = "159985.SZ"
hensheng_tech_etf = "159740.SZ"
gold_etf = "518880.SH"
# 下载518880.SH的30分钟K线数据

df = tf.klines.get(
    hensheng_tech_etf,
    period="30m",
    start_time=start,
    end_time=end,
    as_dataframe=True,
    count=1000
)
# save
# output_file = "Rl_trading_demo/project_one/data/518880.SH.30m_1000.csv"
# timestamp trade_times删除
df = df.drop(columns=['timestamp', 'trade_date'])


# df.to_csv(output_file, index=False)
# print(df.head(20))

print(df.tail(5))
