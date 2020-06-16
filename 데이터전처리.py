#%%
import pandas as pd
from glob import glob
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences

pd.set_option("display.max_columns", 30)
"""
TODO
실제 flux 값을 구해준다. -> B_AVG / 0.7 , A_AVG/0.85
2010-09 ~ 2020-01
B
I < 10 - 6 => 10^-7 까지 하였다. 그 이하는 None 으로 사용
C
10 - 6 < = I < 10 - 5
M
10 - 5 < = I < 10 - 4
X
I > = 10 - 4
등급을 다음과 같이 나눈다
x,m -> 0
c -> 1
b -> 2
none -> 3
"""

# TODO train set 에서 1월이나 10월일 때 Test set 에서 사용한다.
# 그래서 나중에 11월과 12월 Test를 할 때 둘의 차이가 유의미한지 확인이 필요하다.
# 1월에 12월 데이터를 사용하므로 12월의 성능이 확연히 좋다면 데이터를 잘못 사용한 것.


def peak_width_max_mean_std(x, peaks, height):
    results_full = peak_widths(x, peaks, rel_height=height)[0]
    if len(results_full) == 0:
        return 0, 0, 0
    return np.max(results_full), np.mean(results_full), np.std(results_full)


def peak_prominence_max_mean_std(x, peaks):
    prominences = peak_prominences(x, peaks)[0]
    if len(prominences) == 0:
        return 0, 0, 0
    return np.max(prominences), np.mean(prominences), np.std(prominences)


def get_total_data(data, is_B):
    data_num = data.shape[0]
    x = data.values[:, 0].copy()
    if is_B:
        peaks = find_peaks(x, prominence=-7)[0]
    else:
        peaks = find_peaks(x, prominence=-7)[0]
    num_peak = peaks.shape[0]
    max_width3, mean_width3, std_width3 = peak_width_max_mean_std(x, peaks, 0.3)
    peak_data = (
        np.array([num_peak, max_width3, mean_width3, std_width3,]) / data_num
    ).reshape(-1, 1)

    max_pro, mean_pro, std_pro = peak_prominence_max_mean_std(x, peaks)
    pro_data = np.array([max_pro, mean_pro, std_pro]).reshape(-1, 1)
    describe = np.array(
        [np.mean(data.values), np.std(data.values), np.max(data.values),]
    ).reshape(-1, 1)
    total_data = np.concatenate([peak_data, pro_data, describe], axis=0)  # 10,1
    return total_data


def get_xrs_data(
    days, pre_month_days, A_or_B_xrs, year_month, pre_month, is_B, nan_date
):
    day_xrs_seq = []
    for day in pre_month_days:
        data = A_or_B_xrs[pre_month + "-" + str(day)]
        total_data = get_total_data(data, is_B)
        day_xrs_seq.append(total_data)
    for day in days:
        data = A_or_B_xrs[year_month + "-" + str(day)]
        total_data = get_total_data(data, is_B)
        day_xrs_seq.append(total_data)
    day_xrs_seq = np.array(day_xrs_seq)  # n,10,1

    date_range = pd.date_range(
        start=pre_month + "-" + str(pre_month_days[-2]),
        end=year_month + "-" + str(days[-3]),
    )
    date_range_years = date_range.year
    date_range_days = date_range.day
    date_range_months = date_range.month
    day_len = len(days)
    pre_len = len(pre_month_days) - 1  # 전달 마지막날의 전날부터 데이터 수집
    train_xrs_seq = []

    nan_list = []
    for i in range(day_len):
        the_day = f"{date_range_years[i]}-{date_range_months[i]}-{date_range_days[i]}"
        if nan_date[np.isin(nan_date, the_day)].size > 0:
            nan_list.append(i)
            continue
        scope = pre_len + i
        train_xrs_seq.append(day_xrs_seq[scope - 10 : scope])
    train_xrs_seq = np.array(train_xrs_seq).astype(np.float32)  # n,10,10,1

    # 하루전날의 한시간 단위의 데이터를 추가적으로 넣어준다.
    train_xrs_hour_seq = []
    year = pre_month[:4]
    month = int(pre_month[5:])
    for day in pre_month_days[-2:]:
        if nan_date[np.isin(nan_date, f"{year}-{month}-{day}")].size > 0:
            continue
        data = A_or_B_xrs[pre_month + "-" + str(day)].copy()
        min_num = int(data.shape[0] / 24)
        hour_list = []
        for i in range(14, 24):
            hour_data = data[min_num * i : min_num * (i + 1)]
            total_data = get_total_data(hour_data, is_B)
            hour_list.append(total_data)
        train_xrs_hour_seq.append(hour_list)
    year = year_month[:4]
    month = int(year_month[5:])
    for day in days[:-2]:
        if nan_date[np.isin(nan_date, f"{year}-{month}-{day}")].size > 0:
            continue
        data = A_or_B_xrs[year_month + "-" + str(day)].copy()
        min_num = int(data.shape[0] / 24)
        hour_list = []
        for i in range(14, 24):
            hour_data = data[min_num * i : min_num * (i + 1)]
            total_data = get_total_data(hour_data, is_B)
            hour_list.append(total_data)
        train_xrs_hour_seq.append(hour_list)
    train_xrs_hour_seq = np.array(train_xrs_hour_seq).astype(
        np.float32
    )  # day_len,10,10,1
    train_xrs_hour_seq = np.squeeze(train_xrs_hour_seq)
    train_xrs_seq = np.squeeze(train_xrs_seq)
    return day_xrs_seq, train_xrs_seq, train_xrs_hour_seq, nan_list


# 해당 달의 하루 전부터 라벨을 구해준다.
def get_labels(day_xrs_seq, pre_month_days, nan_list, is_train):
    pre_len = len(pre_month_days)
    max_values = day_xrs_seq[pre_len - 1 : -1, -1, :]
    labels = []
    for i, m_val in enumerate(max_values):
        if nan_list[np.isin(nan_list, i)].size > 0:
            continue
        if m_val >= -5:
            labels.append([1.0, 0, 0])
        elif m_val >= -6 and is_train is False:
            labels.append([0, 1.0, 0])
        elif m_val >= -6 and is_train:
            labels.append([0.2, 0.8, 0])
        elif is_train is False:
            labels.append([0, 0, 1.0])
        else:
            labels.append([0, 0.2, 0.8])
    labels = np.array(labels)
    return labels


def min_max_scaler(arr):
    regul_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 0.001)
    return regul_arr


def get_sharp_data(year_month, days, total_sharp, nan_date):
    stand_index = ["ABSNJZH", "SAVNCPP", "TOTPOT", "TOTUSJH", "TOTUSJZ", "USFLUX"]
    month_sharp = []
    for i in range(len(days)):
        from_date = pd.to_datetime(year_month) - pd.Timedelta(11 - i, "d")
        to_date = pd.to_datetime(year_month) - pd.Timedelta(2 - i, "d")
        date = f"{to_date.year}-{to_date.month}-{to_date.day}"
        if nan_date[np.isin(nan_date, date)].size > 0:
            continue
        data = total_sharp[str(from_date)[:10] : str(to_date)[:10]].copy()

        # if data.empty:
        #     print(f"{from_date} ~ {to_date} 완전히 비어있다.")

        top_of_top = pd.DataFrame()

        for col in data.columns:
            data["day"] = data.index.day
            var_top5 = (
                data.sort_values(by=f"{col}", ascending=False)
                .groupby("day")
                .head(5)
                .copy()
                .sort_index()
            ).iloc[:, :-1]
            top_of_top = pd.concat([top_of_top, var_top5], axis=0)
        top_of_top = top_of_top.sort_index().drop_duplicates()
        each_day_counts = []
        date_range = pd.date_range(start=from_date, end=to_date, freq="d")
        for j, month_day in enumerate(date_range):
            day_len = len(top_of_top[month_day : month_day + pd.Timedelta(5, "m")])
            each_day_counts.append(day_len)
        if top_of_top.empty is False:
            abs = min_max_scaler(top_of_top["ABSNJZH"].values)
            sav = min_max_scaler(top_of_top["SAVNCPP"].values)
            totp = min_max_scaler(top_of_top["TOTPOT"].values)
            toth = min_max_scaler(top_of_top["TOTUSJH"].values)
            totz = min_max_scaler(top_of_top["TOTUSJZ"].values)
            usf = min_max_scaler(top_of_top["USFLUX"].values)
            sum_arr = abs + sav + totp + toth + totz + usf
        else:
            sum_arr = []
        counts = 0
        ten_day_df = pd.DataFrame()
        for k, count in enumerate(each_day_counts):
            final_top5 = pd.DataFrame()
            fin_index = np.argsort(sum_arr[counts : count + counts])[-5:]
            for index in fin_index:
                top5 = top_of_top.iloc[counts + index, :]  # 6,
                final_top5 = pd.concat([final_top5, top5], axis=0)  # 6
            if count < 5:
                date = date_range[k]
                zero_df = pd.DataFrame(
                    [-3, -3, -3, -3, -3, -3], index=stand_index, columns=[date]
                )[
                    date
                ]  # log scale 로 했을 때 0이 -6이 된다.
                for _ in range(5 - count):
                    final_top5 = pd.concat([final_top5, zero_df], axis=0)
            # 10,30 -> 10일, 6개의 변수 * 5
            # ex) ABSNJZH -> 0,5,10,15,20  가장 마지막이 가장 값이 높은 top1 값.
            #     SAVNCPP -> 1,6,11,16,21
            # 나중에 각 변수별 데이터를 볼 때 [:,:,i+5*j] 로 보면 된다.
            # 각 날짜 별 & 변수 별은 [:,day,i]
            ten_day_df = pd.concat([ten_day_df, final_top5.transpose()], axis=0)
            counts += count
        month_sharp.append(ten_day_df.values)
    month_sharp = np.array(month_sharp)
    return month_sharp.astype(np.float32)


nan_date = np.load(open("nan_date.npy", "rb"))


xrs_paths = glob("xrs/all/*")[1:-2]
total_sharp = pd.read_pickle("sharp/total_sharp").replace(0, 1e-3)
total_sharp = np.log10(total_sharp)
for xrs_path in xrs_paths:
    year_month = xrs_path[8:]

    if int(year_month[-2:]) < 11:
        is_train = True
    else:
        is_train = False

    print(year_month)
    pre_month = str(pd.to_datetime(year_month) - pd.Timedelta(5, "d"))[:7]

    pre_xrs_data = pd.read_pickle(f"xrs/all/{pre_month}")
    xrs_data = pd.read_pickle(xrs_path)
    total_xrs = pd.concat([pre_xrs_data, xrs_data], axis=0)
    B_xrs = total_xrs[["B_AVG"]].replace(0, 1e-7)
    # A_xrs = total_xrs[["A_AVG"]]

    B_xrs = np.log10(B_xrs)
    # A_xrs = np.log10(A_xrs)
    days = np.unique(xrs_data.index.day)

    pre_month_days = np.unique(pre_xrs_data.index.day)

    day_xrs_seq, B_day_xrs_data, B_hour_xrs_data, nan_list = get_xrs_data(
        days, pre_month_days, B_xrs, year_month, pre_month, is_B=True, nan_date=nan_date
    )  # n,10,10
    nan_list = np.array(nan_list)
    labels = get_labels(day_xrs_seq, pre_month_days, nan_list, is_train)  # n,
    # _, A_day_xrs_data, A_hour_xrs_data, _ = get_xrs_data(
    #     days,
    #     pre_month_days,
    #     A_xrs,
    #     year_month,
    #     pre_month,
    #     is_B=False,
    #     nan_date=nan_date,
    # )  # n,10,10
    # TODO : Sharp 는 데이터가 각각 범위가 달라서 SEQ 를 기준으로 하는 1축으로 Layer Norm 을 해야 한다.
    month_sharp = get_sharp_data(year_month, days, total_sharp, nan_date)  # n,50,6
    if month_sharp.shape[0] == B_day_xrs_data.shape[0] == labels.shape[0]:
        pass
    else:
        print("shape이 다르다.", xrs_path)

    sharp_xrs = np.concatenate(
        [B_day_xrs_data, B_hour_xrs_data, month_sharp], axis=-1,
    )  # n,10,50
    if is_train:
        in_path = "train/"
    else:
        in_path = "test/"
    np.save(open(in_path + f"sharp_xrs/{year_month}.npy", "wb"), sharp_xrs)
    np.save(open(in_path + f"labels/{year_month}.npy", "wb"), labels)

# 대략 3000개 데이터 생성완료.
li = ["sharp_xrs", "labels"]
for t in ["train", "test"]:
    for kind in li:
        peak_paths = glob(f"{t}/{kind}/*")
        all_data = np.load(open(peak_paths[0], "rb"))
        for path in peak_paths[1:]:
            data = np.load(open(path, "rb"))
            all_data = np.concatenate([all_data, data], axis=0)
        np.save(open(f"{t}/{kind}.npy", "wb"), all_data)
        print(f"{kind} - {all_data.shape[0]}개 생성 완료")
