# %%
import matplotlib.font_manager as fm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import seaborn as sns

"""
goes15 데이터는 10년 9월부터 존재한다.
10년 3월 ~ 20년 3월 => 10년치 데이터.
goes10 데이터는 98년 8월 ~ 06년 5월 => 약 8년치 데이터.

To get true fluxes for GOES 8-15 XRS, users must remove the SWPC scaling factors from the data. To
get the true fluxes, divide the short band flux by 0.85 and divide the long band flux by 0.7.
Similarly, to get true fluxes for pre-GOES-8 data, users should divide the archived fluxes by the scale
factors. No adjustments are needed to use the GOES 8-15 fluxes to get the traditional C, M, and X
alert levels.
Class
Peak (W/m2)between 1 and 8 Angstroms
 B
 I < 10-6
 C
 10-6 < = I < 10-5
 M
 10-5 < = I < 10-4
 X
 I > = 10-4


XRS DATA
A : Short
B : Long


SHARP 데이터는 각 흑점에 대한 정보를 보여주는데 흑점이 여러개일 경우
같은 시간에 여러개의 데이터가 존재하게 된다.
이럴경우 한 시간에 대한 맥스값을 사용하거나 해야 할 것 같다.


TODO nan값을 ffill메서드를 사용하여 채워줬는데 그때의 앞 클래스가 뭐였는지
봐야한다. -> mean값을 사용하여 채워줬다.

"""
# #%%
# paths = glob("xrs/xrs_csv/*")
# nan_list = []
# for path in paths:
#     i = 167
#     while True:
#         try:
#             xrs = pd.read_csv(path, skiprows=i)
#             xrs.index = pd.to_datetime(xrs["time_tag"], format="%Y-%m-%d %H:%M:%S")
#             break
#         except:
#             i -= 1
#     xrs["A_AVG"] = xrs["A_AVG"].replace(-99999, np.nan)
#     xrs["B_AVG"] = xrs["B_AVG"].replace(-99999, np.nan)
#     xrs = xrs[["A_AVG", "B_AVG"]]
#     year = xrs.index.year[0]
#     month = xrs.index.month.unique()[0]
#     last_day = xrs.index.day[-1]
#     time_range = pd.date_range(
#         start=str(year) + "-" + str(month) + "-" + str(1),
#         end=str(year) + "-" + str(month) + "-" + str(last_day),
#     )
#     month_df = pd.DataFrame()
#     month_b_mean = xrs["B_AVG"].mean()
#     month_a_mean = xrs["A_AVG"].mean()
#     for day in time_range.day:
#         day_data = xrs[f"{year}-{month}-{day}"]
#         if day_data.empty:
#             day_data = pd.DataFrame(
#                 np.array([np.nan, np.nan] * 1440).reshape(1440, 2),
#                 index=pd.date_range(
#                     start=str(year) + "-" + str(month) + "-" + str(day) + " 00:00:00",
#                     end=str(year) + "-" + str(month) + "-" + str(day) + " 23:59:59",
#                     freq="t",
#                 ),
#                 columns=["A_AVG", "B_AVG"],
#             )
#         drop_data = day_data.dropna()
#         len_day = len(day_data)
#         len_drop = len(drop_data)
#         if len_day - len_drop > 500:
#             nan_list.append(f"{year}-{month}-{day}")
#             a_mean = month_a_mean
#             b_mean = month_b_mean
#         else:
#             a_mean, b_mean = np.mean(day_data)
#
#         xrs_b = day_data[["B_AVG"]].fillna(value=b_mean)
#         xrs_a = day_data[["A_AVG"]].fillna(value=a_mean)
#         xrs_data = pd.concat([xrs_a, xrs_b], axis=1)
#         month_df = pd.concat([month_df, xrs_data], axis=0)
#     month_df.to_pickle(f"xrs/all/{path[12:16]}-{path[16:18]}")
# nan_date = np.array(nan_list)
# nan_list = []
# # 날짜를 사용할 때 결측치가 많은 날을 label 로 쓸 수 없으므로 전날과
# # 다음날의 데이터 예측에 많은 영향을 끼치므로 당일을 제외한다.
# for date in nan_date:
#     nan_list.append(date)
#     date = pd.to_datetime(date)
#     yesterday = date - pd.Timedelta(1, "d")
#     nan_list.append(str(yesterday)[:10])
# np.save(open("nan_date.npy", "wb"), np.unique(nan_list))
#%%
# na = len(xrs.dropna())
# origin = len(xrs)
# if origin - na > 1000:
#     nan_count += 1
#     plt.plot(xrs["B_AVG"])
#     plt.title(f"{path} - {origin-na} 개")
#     plt.xticks(rotation=45)
#     plt.show()
#     plt.close()

# # %%
# for i in range(1, 293):
#     s = pd.read_csv(f"sharp/csv/s ({i}).csv")
#     s.index = pd.to_datetime(s["T_REC"], format="%Y-%m-%d")
#     s = s.sort_index()
#     name = s["T_REC"][0]
#     del s["T_REC"]
#     s.to_pickle(f"sharp/pickle/{name}+{i}")
# a = pd.read_pickle("sharp/pickle/2011-11-15+260")
# b = pd.read_pickle("sharp/pickle/2011-11-15+262")
# # 날짜가 겹치는 것이 있어서 하나로 만들어주고 파일을 삭제하였다.
# pd.concat([a, b], axis=0).to_pickle("sharp/pickle/2011-11-15")
# #%%
# last_month = "0"
# month_df = pd.DataFrame()
# sharp_paths = glob("sharp/pickle/*")
# for path in sharp_paths:
#     year_months = []
#     sharp_data = pd.read_pickle(path)
#     index = sharp_data.index
#     months = np.unique(index.month)
#     years = np.unique(index.year)
#     for year in years:
#         for month in months:
#             try:
#                 year_month = str(year) + "-" + str(month)
#                 sharp_data[year_month]
#                 year_months.append(year_month)
#             except:
#                 pass
#     if last_month == year_months[0]:
#         last_data = pd.read_pickle(f"sharp/data/{last_month}")
#         month_df = sharp_data[last_month]
#         month_df = pd.concat([last_data, month_df], axis=0)
#         month_df.to_pickle(f"sharp/data/{last_month}")
#         year_months = year_months[1:]
#     for year_month in year_months:
#         month_df = sharp_data[year_month]
#         month_df.to_pickle(f"sharp/data/{year_month}")
#
#     if len(year_months) != 0:
#         last_month = year_months[-1]
# sharp_paths = glob("sharp/pickle/*")
# count = 0
# for path in sharp_paths:
#     count += len(pd.read_pickle(path))
# print(count)
# count = 0
# pro_paths = glob("sharp/data/*")
# for path in pro_paths:
#     count += len(pd.read_pickle(path))
# print(count)
# # %%
# xrs_paths = glob('xrs/pickle/*')
# for path in xrs_paths:
#     df = pd.read_pickle(path)
#     name = path[-8:-4]+'-'+path[-4:-2]
#     df.to_pickle(f'xrs/pickle/{name}')
# # %%
# paths = glob("sharp/all/*")
# total_df = pd.DataFrame()
# for path in paths:
#     sharp = pd.read_pickle(path)
#     total_df = pd.concat([total_df, sharp], axis=0)
# total_df.replace(np.inf, 0).to_pickle("sharp/total_sharp")
# #%%
# pd.read_pickle("sharp/total_sharp").describe()
# pre_df.to_pickle("sharp/total_sharp")
#%%
total_df = pd.read_pickle("sharp/total_sharp")
# sharp에 inf 값이 들어있어서 범위를 지정해준다
columns = total_df.columns
plt.figure(figsize=(10, 12))
for i, column in enumerate(columns):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data=total_df, y=f"{column}")
    plt.ylim(np.percentile(total_df[f"{column}"], [0.01, 99.99]))
    plt.title(column)
plt.show()
