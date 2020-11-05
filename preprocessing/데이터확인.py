#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import numpy as np

#%%
# 연도별 분포 확인
xrs_paths = glob("xrs/all/*")
year = "2010"
years = ["2010"]
xcounts = []
mcounts = []
ccounts = []
bcounts = []
xcount = 0
mcount = 0
ccount = 0
bcount = 0
for xrs_path in xrs_paths[:-2]:
    if year != xrs_path[-7:-3]:
        year = xrs_path[-7:-3]
        years.append(year)
        xcounts.append(xcount)
        mcounts.append(mcount)
        ccounts.append(ccount)
        bcounts.append(bcount)
        xcount = 0
        mcount = 0
        ccount = 0
        bcount = 0
    xrs = pd.read_pickle(xrs_path)[["B_AVG"]]
    xrs["time"] = xrs.index.day
    xrs_val = xrs.groupby("time").max().values
    x_num = len(xrs_val[xrs_val > 1e-4])
    xcount += x_num
    m_num = len(xrs_val[xrs_val > 1e-5]) - x_num
    mcount += m_num
    c_num = len(xrs_val[xrs_val > 1e-6]) - x_num - m_num
    ccount += c_num
    bcount += len(xrs_val) - x_num - m_num - c_num

xcounts.append(xcount)
mcounts.append(mcount)
ccounts.append(ccount)
bcounts.append(bcount)
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
df = pd.DataFrame(xcounts, index=years, columns=["xcounts"])
sns.barplot(data=df, x=df.index, y="xcounts")
plt.ylim(0, 300)
plt.title("X RANK")
plt.subplot(2, 2, 2)
df = pd.DataFrame(mcounts, index=years, columns=["mcounts"])
sns.barplot(data=df, x=df.index, y="mcounts")
plt.title("M RANK")
plt.ylim(0, 300)
plt.subplot(2, 2, 3)
df = pd.DataFrame(ccounts, index=years, columns=["ccounts"])
sns.barplot(data=df, x=df.index, y="ccounts")
plt.title("C RANK")
plt.ylim(0, 300)
plt.subplot(2, 2, 4)
df = pd.DataFrame(bcounts, index=years, columns=["bcounts"])
sns.barplot(data=df, x=df.index, y="bcounts")
plt.title("B RANK")
plt.ylim(0, 300)
plt.show()
plt.close()
#%%
# 연도별 분포 확인
xrs_paths = glob("xrs/all/*")
year = "2010"
years = ["2010"]
xcounts = []
ccounts = []
bcounts = []
ncounts = []
xcount = 0
ccount = 0
bcount = 0
ncount = 0
for xrs_path in xrs_paths[:-2]:
    if year != xrs_path[-7:-3]:
        year = xrs_path[-7:-3]
        years.append(year)
        xcounts.append(xcount)
        ccounts.append(ccount)
        bcounts.append(bcount)
        ncounts.append(ncount)
        xcount = 0
        ccount = 0
        bcount = 0
        ncount = 0
    xrs = pd.read_pickle(xrs_path)[["B_AVG"]]
    xrs["time"] = xrs.index.day
    xrs_val = xrs.groupby("time").max().values
    x_num = len(xrs_val[xrs_val > 1e-5])
    xcount += x_num
    c_num = len(xrs_val[xrs_val > 1e-6]) - x_num
    ccount += c_num
    b_num = len(xrs_val[xrs_val > 1e-7]) - x_num - c_num
    bcount += b_num
    ncount += len(xrs_val) - x_num - c_num - b_num

xcounts.append(xcount)
ccounts.append(ccount)
bcounts.append(bcount)
ncounts.append(ncount)
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
df = pd.DataFrame(xcounts, index=years, columns=["xcounts"])
sns.barplot(data=df, x=df.index, y="xcounts")
plt.ylim(0, 300)
plt.title("X_M RANK")
plt.subplot(2, 2, 2)
df = pd.DataFrame(ccounts, index=years, columns=["ccounts"])
sns.barplot(data=df, x=df.index, y="ccounts")
plt.title("C RANK")
plt.ylim(0, 300)
plt.subplot(2, 2, 3)
df = pd.DataFrame(bcounts, index=years, columns=["bcounts"])
sns.barplot(data=df, x=df.index, y="bcounts")
plt.title("B RANK")
plt.ylim(0, 300)
plt.subplot(2, 2, 4)
df = pd.DataFrame(ncounts, index=years, columns=["ncounts"])
sns.barplot(data=df, x=df.index, y="ncounts")
plt.title("N RANK")
plt.ylim(0, 300)
plt.show()
plt.close()
#%%
# train,test 나눠서 확인
xrs_paths = glob("xrs/all/*")
train_xcounts = 0
train_ccounts = 0
train_bcounts = 0
train_ncounts = 0
test_xcounts = 0
test_ccounts = 0
test_bcounts = 0
test_ncounts = 0
for xrs_path in xrs_paths[:-2]:
    if int(xrs_path[-2:]) < 11:
        xrs = pd.read_pickle(xrs_path)[["B_AVG"]]
        xrs["time"] = xrs.index.day
        xrs_val = xrs.groupby("time").max().values
        x_num = len(xrs_val[xrs_val > 1e-5])
        train_xcounts += x_num
        c_num = len(xrs_val[xrs_val > 1e-6]) - x_num
        train_ccounts += c_num
        b_num = len(xrs_val[xrs_val > 1e-7]) - x_num - c_num
        train_bcounts += b_num
        train_ncounts += len(xrs_val) - x_num - c_num - b_num
    else:
        xrs = pd.read_pickle(xrs_path)[["B_AVG"]]
        xrs["time"] = xrs.index.day
        xrs_val = xrs.groupby("time").max().values
        x_num = len(xrs_val[xrs_val > 1e-5])
        test_xcounts += x_num
        c_num = len(xrs_val[xrs_val > 1e-6]) - x_num
        test_ccounts += c_num
        b_num = len(xrs_val[xrs_val > 1e-7]) - x_num - c_num
        test_bcounts += b_num
        test_ncounts += len(xrs_val) - x_num - c_num - b_num
df = pd.DataFrame(
    [train_xcounts, train_ccounts, train_bcounts, train_ncounts],
    index=["x_m", "c", "b", "n"],
    columns=["train_counts"],
)
sns.barplot(data=df, x=df.index, y="train_counts")
plt.title("Train Set")
plt.show()
df = pd.DataFrame(
    [test_xcounts, test_ccounts, test_bcounts, test_ncounts],
    index=["x", "c", "b", "n"],
    columns=["test_counts"],
)
sns.barplot(data=df, x=df.index, y="test_counts")
plt.title("Test Set")
plt.show()
#%%
group_colors = ["yellowgreen", "lightskyblue", "lightcoral", "gray"]
group_explodes = (0.1, 0, 0, 0)  # explode 1st slice
group_names = ["X_M Rank", "C Rank", "B Rank", "N Rank"]
plt.pie(
    [train_xcounts, train_ccounts, train_bcounts, train_ncounts],
    explode=group_explodes,
    colors=group_colors,
    labels=group_names,
    autopct="%1.2f%%",
    startangle=90,
    textprops={"fontsize": 14},
    shadow=True,
)
plt.title("Pie Chart of Train Set", fontsize=20)
plt.axis("equal")
plt.show()

group_colors = ["yellowgreen", "lightskyblue", "lightcoral", "gray"]
group_explodes = (0.1, 0, 0, 0)  # explode 1st slice
group_names = ["X_M Rank", "C Rank", "B Rank", "N Rank"]
plt.pie(
    [test_xcounts, test_ccounts, test_bcounts, test_ncounts],
    explode=group_explodes,
    colors=group_colors,
    labels=group_names,
    autopct="%1.2f%%",
    startangle=90,
    textprops={"fontsize": 14},
    shadow=True,
)
plt.title("Pie Chart of Test Set", fontsize=20)
plt.axis("equal")
plt.show()
