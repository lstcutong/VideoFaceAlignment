import numpy as np
import matplotlib.pyplot as plt

COLORS = [
    "#ED5565",
    "#FFCE54",
    "#48CFAD",
    "#5D9CEC",
    "#EC87C0",
    "#CCD1D9",
    "#A0D468",
    "#4FC1E9",
    "#AC92EC"
]


def string2hex(string):
    hex = 0
    for i in range(len(string)):
        if string[i] in ["{}".format(i) for i in range(10)]:
            hex += int(string[i]) * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "A":
            hex += 10 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "B":
            hex += 11 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "C":
            hex += 12 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "D":
            hex += 13 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "E":
            hex += 14 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "F":
            hex += 15 * 16 ** (len(string) - i - 1)
    return int(hex)


def string2rgb(string):
    r, g, b = string[1:3], string[3:5], string[5:7]
    return (string2hex(r), string2hex(g), string2hex(b),)


def calculate_NME(pred_landmarks, gt_landmarks, bbox):
    '''

    :param pred_landmarks:  [seq_len, 68, 2]
    :param gt_landmarks:[seq_len, 68, 2]
    :param bbox:
    :return: se [seq_len,] nse double
    '''
    d = np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    se = np.mean(np.sqrt(np.sum((pred_landmarks - gt_landmarks) ** 2, axis=2)) / d, axis=1)

    nse = np.mean(se)

    return se, nse


def draw_metrics(se_lists, names, title, save_path):
    plt.clf()
    fig = plt.figure(figsize=(20,3))
    for i in range(len(se_lists)):
        seq_len = len(se_lists[i])

        x = np.arange(seq_len)
        y = se_lists[i]

        plt.plot(x, y, color=COLORS[i], label=names[i])

    plt.title(title)
    plt.xlabel("sequence number")
    plt.ylabel("nme")
    plt.legend()
    plt.show()
    plt.savefig(save_path)
