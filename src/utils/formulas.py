import numpy as np


# Reference: https://archimer.ifremer.fr/doc/00287/39795/94062.pdf, page 10
def calc_do_saturation(temp, sal, umol_L=False):
    A0 = 2.00907
    A1 = 3.22014
    A2 = 4.0501
    A3 = 4.94457
    A4 = -0.256847
    A5 = 3.88767

    B0 = -0.00624523
    B1 = -0.00737614
    B2 = -0.0103410
    B3 = -0.00817083
    C0 = -0.00000048868

    temp_kelvin = np.log((298.15 - temp) / (273.15 + temp))

    # mL/L
    do_saturation = np.exp(
        A0
        + A1 * temp_kelvin
        + A2 * temp_kelvin**2
        + A3 * temp_kelvin**3
        + A4 * temp_kelvin**4
        + A5 * temp_kelvin**5
        + sal * (B0 + B1 * temp_kelvin + B2 * temp_kelvin**2 + B3 * temp_kelvin**3)
        + C0 * sal**2
    )

    if umol_L:
        return do_saturation * 44.6596
    else:
        return do_saturation
