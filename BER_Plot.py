import numpy as np
import matplotlib.pyplot as plt 
from scipy import special
import os

PATH = 'plots'

os.chdir(os.path.join(os.path.dirname(__file__), PATH))

M = 2
bits_per_symbol = int(np.log2(M))
num_data_symbols = 10000
num_data_bits = bits_per_symbol * num_data_symbols

# Generating bit stream from 0 to M-th symbols
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

def AWGN(input, sigma=1.0):
    N = input.size
    return np.real(input) + np.random.normal(0.0, sigma, N) + 1j * (np.imag(input) + np.random.normal(0.0, sigma, N))

sigma_range = 10**(np.arange(-2, 0, 0.1))
EbN0 = np.zeros(sigma_range.size)
EbN0_dB = np.zeros(sigma_range.size)
bit_error_rate = np.zeros(sigma_range.size)

if M == 2 or M == 4:
    max_symbol_val = 1
elif M == 16:
    max_symbol_val = 3

for k, sigma in enumerate(sigma_range):
    mqam_symbols_noise = AWGN(mqam_symbols, sigma) # Add noise to symbols

    symbol_power = 1.0  # Normalized symbol power
    bit_power = symbol_power / np.log2(M)
    noise_power = 2 * sigma**2  # Complex Gaussian

    EbN0[k] = bit_power / noise_power
    EbN0_dB[k] = 10 * np.log10(EbN0[k])

    # Demodulate symbols and compute BER
    demod_bits = np.zeros((mqam_symbols_noise.size, bits_per_symbol), dtype=np.int32)
    for sym in range(mqam_symbols_noise.size):
        symbol = mqam_symbols_noise[sym] * mqamNormFactor
        real = np.clip(np.round((symbol.real + 1) / 2, 0) * 2 - 1, -max_symbol_val, max_symbol_val)
        if M == 2:
            symbol = real
        else:
            imag = np.clip(np.round((symbol.imag + 1) / 2, 0) * 2 - 1, -max_symbol_val, max_symbol_val)
            symbol = real + 1j * imag
        demod_bits[sym] = [int(bit) for bit in reversed(MEncodingRev[M][symbol])]

    demod_bits = demod_bits.flatten()

    bit_error_rate[k] = np.sum(np.abs(bits - demod_bits)) / bits.size

def Q(x):
    return 0.5 * special.erfc(x / np.sqrt(2))

bitSNR_dB = np.arange(0, 22, 0.1)
bitSNR = 10**(bitSNR_dB / 10)

def MQ(x, M, N0=1):
    nBits = np.log2(M)
    return 4 / nBits * Q(np.sqrt(3 * nBits * x / (N0 * (M - 1))))

Q_BPSK_QAM4 = Q(np.sqrt(2 * bitSNR))
Q_QAM16 = MQ(bitSNR, 16)
Q_QAM64 = MQ(bitSNR, 64)

plt.figure()
plt.semilogy(bitSNR_dB, Q_BPSK_QAM4, label='BPSK/4-QAM')
plt.semilogy(bitSNR_dB, Q_QAM16, label='16-QAM')
plt.semilogy(bitSNR_dB, Q_QAM64, label='64-QAM')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER Curve for Different Modulations (M={})'.format(M))
plt.xlim([0, 22])
plt.ylim([10**-8, 1])
plt.grid(True)
plt.legend()

# Overlay simulated BER points on theoretical curve
plt.semilogy(EbN0_dB, bit_error_rate, 'x')

plt.savefig('BER_Q_Funcs_M{}.png'.format(M))