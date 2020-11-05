# %%
from selenium import webdriver
from bs4 import BeautifulSoup
import re
from time import sleep
import numpy as np
import os
from glob import glob
# %%
# 데이터를 받는다. 10년 9월부터 20년 3월까지
a = np.arange(1, 10)
month = []
for m in a:
    month.append('0'+str(m))
month.append('10')
month.append('11')
month.append('12')
year = np.arange(2010, 2021)
driver = webdriver.Chrome("chromedriver.exe")
for y in year:
    for m in month:
        try:
            url = f'https://satdat.ngdc.noaa.gov/sem/goes/data/avg/{y}/{m}/goes15/csv/'

            driver.get(url)
            bs = BeautifulSoup(driver.page_source, 'lxml')
            bs_a = bs.find_all('a')
            for a in bs_a:
                g_15 = re.search('g15_xrs_1m_[0-9]+_[0-9]+.csv', str(a))
                if g_15 is not None:
                    break
            params = g_15.group()
            driver.get(url+params)
        except Exception as e:
            print(y, m)
            print(e)
# %%
# 받은 데이터를 옮긴다.
file_paths = glob('C:/Users/brian/Downloads/*.csv')
for path in file_paths:
    try:
        os.rename(f'{path}',
                  f'xrs/{path[-21:]}')
    except:
        print(path)
