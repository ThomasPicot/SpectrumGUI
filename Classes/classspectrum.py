import numpy as np
import scipy.constants as cst
from scipy.special import wofz


class SpectrumRubidiumD2Line:
    def __init__(self, gamma: float = 3.81138e7, k: float = 2 * np.pi / 7.807864080702083e-7,
                 d: float = 4.4003140382156403e-29, M: float = 1.4099931997e-25, beta: float = 1.03e-13):
        """

        Parameters
        ----------
        gamma : decay time for the stage |e> -> |g>
        k : laser wave number
        d : electric dipole
        M : atomic mass
        beta : [SI] impact self broadening
        """
        self.h = cst.hbar
        self.eps_zero = cst.epsilon_0
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.d = d
        self.M = M

        # Rb 87

        # F_g = 2 --> F_e = 1, 2, 3
        self.w_i = -2735.05e6 + 1307.87e6
        self.w_ii = -2578.11e6 + 1307.87e6
        self.w_iii = -2311.26e6 + 1307.87e6

        # F_g = 1 --> F_e = 0, 1, 2
        self.w_ib = 4027.403e6 + 1307.87e6
        self.w_iib = 4099.625e6 + 1307.87e6
        self.w_iiib = 4256.57e6 + 1307.87e6

        # Rb 85

        # F_g = 3 --> F_e = 2, 3, 4
        self.w_j = -1371.29e6 + 1307.87e6
        self.w_jj = -1307.87e6 + 1307.87e6
        self.w_jjj = -1186.91e6 + 1307.87e6

        # F_g = 2 --> F_e = 1, 2, 3
        self.w_jb = 1635.454e6 + 1307.87e6
        self.w_jjb = 1664.714e6 + 1307.87e6
        self.w_jjjb = 1728.134e6 + 1307.87e6

    @staticmethod
    def voigtProfile(z: float) -> float:
        """
        function that has a gaussian profile and that takes the effect of doppler broadening.
        :param z: function you want to apply the voigt profile
        :return: non-complex number voigt profile
        """
        return 1j * wofz(1j * z)

    @staticmethod
    def Ndensity(temp: int, deg: int) -> float:
        """
        calculates the atomic density, depending on the isotope and temperature.
        :param temp: temperature T[K] (local var)
        :param deg: degeneration degree, depending of the isotope. 8 for 87 and 12 for 85
        :return: density N depending of the degeneration
        """
        N = (133.323 * 10 ** (15.88253 - (4529.635 / temp) + 0.00058663 * temp - 2.99138 * np.log10(temp))) / (
                cst.Boltzmann * temp)
        return N / deg

    @staticmethod
    def sigma(temp: int) -> float:
        """
        width of the velocity distribution
        :param temp: temperature T[K] (local var)
        :return: sigma
        """
        sigma: float = (2 * np.pi/ 7.807864080702083e-7) * np.sqrt((2 * cst.k * temp) / 1.4099931997e-25)
        return sigma

    def detuning(self) -> np.array([float]):
        """
        variable
        :return: detuning array
        """
        detuning: array = np.linspace(-7e9, 7e9, 10000)  # useless to touch it
        return detuning

    def atomNumber(self, frac: int, temp: int, deg: int) -> float:
        """
        :param frac:
        :param temp:
        :param deg:
        :return: N the mean atom number contained in m^3
        """
        N = frac * self.Ndensity(temp, deg)
        return N

    def susceptibility(self, C_f: float, frac: int, temp: int, transition_frequency: float,
                       deg: int) -> np.array([float]):
        """

        :param C_f: C-f^2 transition coefficient, depending on the quantum number F and isotope
        :param frac: % of Rb 87 in the cell. frac in [0-100]
        :param temp: temperature T[K] (local var)
        :param transition_frequency: possible values are instance variables in init that depend of the quantum number F.
        :param deg: degeneration degree, depending of the isotope. 8 for 87 and 12 for 85
        :return: absorption coefficient
        """
        N: float = self.atomNumber(frac, temp, deg)
        delta: np.array([int]) = 2 * np.pi * (self.detuning() - transition_frequency)
        voigt_arg = (0.5*(self.gamma+2*np.pi*self.beta*self.Ndensity(temp=temp, deg=deg)) - 1j * delta) / self.sigma(temp)
        suscepti = C_f * (N * (self.d ** 2) * np.sqrt(np.pi) / (self.h * self.eps_zero * self.sigma(temp))) * \
                   self.voigtProfile(voigt_arg).imag
        return suscepti

    def alpha(self, C_f, frac, temp, transition_frequency, deg) -> np.array([float]):
        """
        absorption coefficient alpha
        :param C_f:
        :param frac:
        :param temp:
        :param transition_frequency:
        :param deg:
        :return:
        """
        alpha = self.k * self.susceptibility(C_f, frac, temp, transition_frequency, deg)
        return alpha

    def transmission(self, frac=None, temp: int = 300, long: float = 0.075) -> np.array([float]):
        sum_alpha = self.alpha(10 / 81, 1 - (frac / 100), temp, self.w_j, 12) + \
                    self.alpha(35 / 81, 1 - (frac / 100), temp, self.w_jj, 12) + \
                    self.alpha(1, 1 - (frac / 100), temp, self.w_jjj, 12) + \
                    self.alpha(1 / 3, 1 - (frac / 100), temp, self.w_jb, 12) + \
                    self.alpha(35 / 81, 1 - (frac / 100), temp, self.w_jjb, 12) + \
                    self.alpha(28 / 81, 1 - (frac / 100), temp, self.w_jjjb, 12) + \
                    self.alpha(1 / 18, frac / 100, temp, self.w_i, 8) + \
                    self.alpha(5 / 18, frac / 100, temp, self.w_ii, 8) + \
                    self.alpha(7 / 9, frac / 100, temp, self.w_iii, 8) + \
                    self.alpha(1 / 9, frac / 100, temp, self.w_ib, 8) + \
                    self.alpha(5 / 18, frac / 100, temp, self.w_iib, 8) + \
                    self.alpha(5 / 18, frac / 100, temp, self.w_iiib, 8)
        transmi = np.exp(-sum_alpha * long)
        return transmi
