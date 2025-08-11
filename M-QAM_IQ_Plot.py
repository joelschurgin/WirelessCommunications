import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import toeplitz
from rrcosfilter import *
import os

PATH = 'plots'

os.chdir(os.path.join(os.path.dirname(__file__), PATH))

M = 16
bits_per_symbol = int(np.log2(M))
num_data_symbols = M
num_data_bits = bits_per_symbol * num_data_symbols

decimals = np.arange(num_data_symbols)
bits = np.zeros(num_data_symbols * bits_per_symbol, dtype=int)

for sym in range(num_data_symbols):
    for b in range(bits_per_symbol):
        bits[b + sym * bits_per_symbol] = (decimals[sym] >> b) & 0x1

symbols = bits.reshape(num_data_symbols, bits_per_symbol)
mqam_symbols = np.zeros(num_data_symbols, dtype=np.complex128)

M2 = {'0': -1, '1': 1}
M4 = {'00': -1-1j, '01': 1-1j, '10': -1+1j, '11': 1+1j}
M16 = {'0000': -3-3j, '0001': -1-3j, '0011': 1-3j, '0010': 3-3j,
       '0100': -3-1j, '0101': -1-1j, '0111': 1-1j, '0110': 3-1j,
       '1100': -3+1j, '1101': -1+1j, '1111': 1+1j, '1110': 3+1j,
       '1000': -3+3j, '1001': -1+3j, '1011': 1+3j, '1010': 3+3j}
MEncoding = {2: M2, 4: M4, 16: M16}
MEncodingRev = {2: {k:v for v,k in M2.items()}, 
                4: {k:v for v,k in M4.items()},
                16: {k:v for v,k in M16.items()}}

mqamNormFactor = 1
if M == 4:
    mqamNormFactor = np.sqrt(2)
elif M == 16:
    mqamNormFactor = np.sqrt(10)

for sym in range(num_data_symbols):
    sym_str = ''
    for bit in reversed(symbols[sym]):
        sym_str += str(bit)
    mqam_symbols[sym] = MEncoding[M][sym_str] / mqamNormFactor

# Plot M-QAM symbols
plt.figure()
plt.plot(np.real(mqam_symbols), np.imag(mqam_symbols), 'o')
plt.xlabel('I')
plt.ylabel('Q')
plt.grid(True)
# plt.legend()
ax = plt.gca()
ax.set_aspect('equal', adjustable='datalim')
for sym in range(M):
    # This code will plot the binary code together
    plt.text(np.real(mqam_symbols[sym]), np.imag(mqam_symbols[sym]), bin(sym)[2:].zfill(bits_per_symbol))
plt.savefig('{}-QAM_IQ_Plot.png'.format(M))

demod_bits = np.zeros((mqam_symbols.size, bits_per_symbol), dtype=np.int32)
for sym in range(mqam_symbols.size):
    symbol = mqam_symbols[sym] * mqamNormFactor
    demod_bits[sym] = [int(bit) for bit in reversed(MEncodingRev[M][symbol])]

demod_bits = demod_bits.flatten()

# BER Calculation
print('Transmitted bits:', bits)
print('Demodulated bits:', demod_bits)
bit_err_rate = np.sum(np.abs(bits - demod_bits)) / bits.size
print('BER:', bit_err_rate)