import numpy as np


def pm25_iaqi(row: np.array):
    if row[0] >= 350:
        row[0] = 400 + (row[0] - 350) * (500 - 400) / (500 - 350)
    elif row[0] >= 250:
        row[0] = 300 + (row[0] - 250) * (400 - 300) / (350 - 250)
    elif row[0] >= 150:
        row[0] = 200 + (row[0] - 150) * (300 - 200) / (250 - 150)
    elif row[0] >= 115:
        row[0] = 150 + (row[0] - 115) * (200 - 150) / (150 - 115)
    elif row[0] >= 75:
        row[0] = 100 + (row[0] - 75) * (150 - 100) / (115 - 75)
    elif row[0] >= 35:
        row[0] = 50 + (row[0] - 35) * (100 - 50) / (75 - 35)
    else:
        row[0] = row[0] * 50 / 35
    return row


def pm10_iaqi(row: np.array):
    if row[1] >= 500:
        row[1] = 400 + (row[1] - 500) * (500 - 400) / (600 - 500)
    elif row[1] >= 420:
        row[1] = 300 + (row[1] - 420) * (400 - 300) / (500 - 420)
    elif row[1] >= 350:
        row[1] = 200 + (row[1] - 350) * (300 - 200) / (420 - 350)
    elif row[1] >= 250:
        row[1] = 150 + (row[1] - 250) * (200 - 150) / (350 - 250)
    elif row[1] >= 150:
        row[1] = 100 + (row[1] - 150) * (150 - 100) / (250 - 150)
    elif row[1] >= 50:
        row[1] = 50 + (row[1] - 50) * (100 - 50) / (150 - 50)
    else:
        row[1] = row[1] * 50 / 50
    return row


def o3_iaqi(row: np.array):
    if row[2] >= 400:
        row[2] = 200 + (row[2] - 400) * (300 - 200) / (800 - 400)
    elif row[2] >= 300:
        row[2] = 150 + (row[2] - 300) * (200 - 150) / (400 - 300)
    elif row[2] >= 200:
        row[2] = 100 + (row[2] - 200) * (150 - 200) / (300 - 200)
    elif row[2] >= 160:
        row[2] = 50 + (row[2] - 160) * (100 - 50) / (200 - 160)
    else:
        row[2] = row[2] * 50 / 160
    return row


def so2_iaqi(row: np.array):
    if row[3] >= 650:
        row[3] = 150 + (row[3] - 650) * (200 - 150) / (800 - 650)
    elif row[3] >= 500:
        row[3] = 100 + (row[3] - 500) * (150 - 100) / (650 - 500)
    elif row[3] >= 150:
        row[3] = 50 + (row[3] - 150) * (100 - 50) / (500 - 150)
    else:
        row[3] = row[3] * 50 / 150
    return row


def no2_iaqi(row: np.array):
    if row[4] >= 3090:
        row[4] = 400 + (row[4] - 3090) * (500 - 400) / (3840 - 3090)
    elif row[4] >= 2340:
        row[4] = 300 + (row[4] - 2340) * (400 - 300) / (3090 - 2340)
    elif row[4] >= 1200:
        row[4] = 200 + (row[4] - 1200) * (300 - 200) / (2340 - 1200)
    elif row[4] >= 700:
        row[4] = 150 + (row[4] - 700) * (200 - 150) / (1200 - 700)
    elif row[4] >= 200:
        row[4] = 100 + (row[4] - 200) * (150 - 100) / (700 - 200)
    elif row[4] >= 100:
        row[4] = 50 + (row[4] - 100) * (100 - 50) / (200 - 100)
    else:
        row[4] = row[4] * 50 / 100
    return row


def co_iaqi(row: np.array):
    if row[5] >= 120:
        row[5] = 400 + (row[5] - 120) * (500 - 400) / (150 - 120)
    elif row[5] >= 90:
        row[5] = 300 + (row[5] - 90) * (400 - 300) / (120 - 90)
    elif row[5] >= 60:
        row[5] = 200 + (row[5] - 60) * (300 - 200) / (90 - 60)
    elif row[5] >= 35:
        row[5] = 150 + (row[5] - 35) * (200 - 150) / (60 - 35)
    elif row[5] >= 10:
        row[5] = 100 + (row[5] - 10) * (150 - 100) / (35 - 10)
    elif row[5] >= 5:
        row[5] = 50 + (row[5] - 5) * (100 - 50) / (10 - 5)
    else:
        row[5] = row[5] * 50 / 5
    return row


def cal_aqi(ac: np.array) -> np.array:
    shape = ac.shape
    ac = ac.reshape(-1, shape[-1])

    iaqi = np.copy(ac)

    for f in [pm25_iaqi, pm10_iaqi, o3_iaqi, so2_iaqi, no2_iaqi, co_iaqi]:
        np.apply_along_axis(f, 0, iaqi)

    aqi = np.max(iaqi, axis=1)
    return aqi.reshape(*shape[:-1])


if __name__ == '__main__':
    # TODO: verify `cal_aqi()`
    print(cal_aqi(np.random.randn(10, 11, 12, 13, 5)).shape)
