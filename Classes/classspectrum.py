import numpy as np
import scipy.constants as cst
from scipy.special import wofz


# voigt function
def V(z):
    return 1j*wofz(1j*z)


def Ndensity(temp, deg):
    N = (133.323 * 10 ** (15.88253 - (4529.635 / temp) + 0.00058663 * temp - 2.99138 * np.log10(temp))) / (
                cst.Boltzmann * temp)
    return N/deg


# change the temperature parameter in the function
def varsigma(temp):
    sigma = 2 * np.pi * np.sqrt((2 * cst.k * temp) / (1.4099931997e-25 * 7.807864080702083e-7 ** 2))
    return sigma


class DisplaySpectrum:
    def __init__(self, gamma=0.5 * 3.81138e7, k=2*np.pi/7.807864080702083e-7, v=2e3, m=1.4099931997e-25):
        self.h = cst.hbar
        self.eps_zero = cst.epsilon_0
        self.gamma = gamma
        self.k = k
        self.d = 3.24627e-29
        self.v = v
        self.m = m

        # Rb 87

        # F_g = 2 --> F_e = 1, 2, 3
        self.w_i = -2735.05e6
        self.w_ii = -2578.11e6
        self.w_iii = -2311.26e6

        # F_g = 1 --> F_e = 0, 1, 2
        self.w_ib = 4027.403e6
        self.w_iib = 4099.625e6
        self.w_iiib = 4256.57e6

        # Rb 85

        # F_g = 3 --> F_e = 2, 3, 4
        self.w_j = -1371.29e6
        self.w_jj = -1307.87e6
        self.w_jjj = -1186.91e6

        # F_g = 2 --> F_e = 1, 2, 3
        self.w_jb = 1635.454e6
        self.w_jjb = 1664.714e6
        self.w_jjjb = 1728.134e6

    def varomega(self):
        omega = 2*np.pi*np.linspace(-5e9, 5e9, 3000)     # useless to touch it
        return omega

    def fracRb(self, frac, temp, deg):
        N = frac*Ndensity(temp, deg)
        return N

    def absorption(self, C_f, frac, temp, detuning, deg):
        N = self.fracRb(frac, temp, deg)
        delta = 2*np.pi*(self.varomega() - detuning)
        voigt_arg = (self.gamma - 1j * delta) / varsigma(temp)
        absorption = C_f*(N * (self.d ** 2) * np.sqrt(np.pi) / (self.h * self.eps_zero *varsigma(temp))) * V(
            voigt_arg).imag
        return absorption

    def alpha(self, C_f, frac, temp, detuning, deg):
        alpha = self.k*self.absorption(C_f, frac, temp, detuning, deg)
        return alpha

    def transmission(self, frac=None, temp=300, long=0.075):
        sum_alpha = self.alpha(10/81, 1-(frac/100), temp, self.w_j, 12) + self.alpha(35/81, 1-(frac/100),
                                                                                     temp, self.w_jj, 12) + \
                    self.alpha(1, 1-(frac/100), temp, self.w_jjj, 12) + self.alpha(1/3, 1-(frac/100), temp,
                                                                                   self.w_jb, 12) + \
                    self.alpha(35/81, 1-(frac/100), temp, self.w_jjb, 12) + self.alpha(28/81, 1-(frac/100), temp,
                                                                                       self.w_jjjb, 12) +\
                    self.alpha(1/18, frac/100, temp, self.w_i, 8) + self.alpha(5/18, frac/100, temp, self.w_ii, 8) +\
                    self.alpha(7/9, frac/100, temp, self.w_iii, 8) + self.alpha(1/9, frac/100, temp, self.w_ib, 8) +\
                    self.alpha(5/18, frac/100, temp, self.w_iib, 8) + self.alpha(5/18, frac/100, temp, self.w_iiib, 8)
        transmi = np.exp(-sum_alpha * long)
        return transmi
