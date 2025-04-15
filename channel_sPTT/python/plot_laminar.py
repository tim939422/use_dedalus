import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature'])
import numpy as np

if __name__ == '__main__':
    data_morozov = np.loadtxt('prl_2022_Morozov_Fig2b.txt')
    data_zero_diffusion = np.loadtxt('zero_diffusion.txt')
    pass