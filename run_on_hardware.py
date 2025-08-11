import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
import adi
from rrcosfilter import *
import os

PATH = 'plots'

os.chdir(os.path.join(os.path.dirname(__file__), PATH))

num_symbols = 16
sps = 32
fs = 1e6
Ns = 1000
carrier_freq = 2e9
alpha = 1
L = 4

M = 16

bits_per_symbol = int(np.log2(M))
num_data_bits = bits_per_symbol * num_symbols

# setup PlutoSDR
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(fs)
sdr.tx_rf_bandwidth = int(fs) 
sdr.tx_lo = int(carrier_freq)
sdr.tx_hardwaregain_chan0 = -50
sdr.rx_lo = int(carrier_freq)
sdr.rx_rf_bandwidth = int(fs)
sdr.rx_buffer_size = Ns
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 30.0

bits = np.random.randint(0, 2, num_data_bits) # random integers``, 0 or 1

# Add preamble and guard band to create a frame
preamble = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1])
guard_band = np.zeros(preamble.shape)

symbols = bits.reshape(num_symbols, bits_per_symbol)
mqam_symbols = np.zeros(num_symbols, dtype=np.complex128)

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

for sym in range(num_symbols):
    sym_str = ''
    for bit in reversed(symbols[sym]):
        sym_str += str(bit)
    mqam_symbols[sym] = MEncoding[M][sym_str] / mqamNormFactor

tx_symbols = np.concatenate((preamble, preamble, guard_band, mqam_symbols))

# Upsampling and pulse-shaping from the previous labs
up_sym = np.zeros(tx_symbols.shape[0] * sps, dtype=np.complex128)
for i in range(tx_symbols.shape[0]):
    up_sym[i * sps] = tx_symbols[i]

rect = np.ones(sps) / sps
postConv = np.convolve(up_sym, rect)

time_idx, h_rrc = rrcosfilter(9*sps+1, alpha, 1, sps)
pulseShaped = np.convolve(postConv, h_rrc)

# Modulated CPFSK samples
tx_samples = pulseShaped * 2**14

# Transmit code from the previous part
# Start transmitting
sdr.tx_cyclic_buffer = True 
sdr.tx(tx_samples)

# Code omitted for clearing buffer and stopping transmission
# Set Rx buffer size to 2x
sdr.rx_buffer_size = 2 * tx_samples.size

# Clear buffer just to be safe
for i in range (0, 5):
    raw_data = sdr.rx()

raw_data = sdr.rx()
rx_samples = 2**-14*raw_data

# Stop transmitting
sdr.tx_destroy_buffer()

# Apply the matched filter
match_filter = rrcosfilter(5*sps+1, alpha, 1, sps)[1]
rx_matched = np.convolve(rx_samples, match_filter)

# Symbol Timimg Recovery
energy = np.zeros(sps)
for k in range(sps):
    energy[k] = sum(rx_matched[n * sps + k].real**2 + rx_matched[n * sps + k].imag**2 for n in range(int(len(rx_matched) / sps)))

# align the samples
max_ind = np.argmax(energy)
rx_align = np.roll(rx_matched, -max_ind)
rx_cfo = rx_align[::sps]

# Self-reference frame sync
N = preamble.size
corr = np.zeros(rx_cfo.size - 2 * N + 1)
for k in range(len(corr)):
    corr[k] = np.abs(np.sum(np.conjugate(rx_cfo[k:k+N]) * rx_cfo[k+N:k+N*2], dtype=np.complex64))  # Add your correlation calculation here

# Use find_peaks to detect the right peak
peak_idxs = find_peaks(corr, prominence=np.max(corr) * 0.75)[0]
first_peak = peak_idxs[0]

# Find out the frame, then split it into preamble and data
rx_frame = rx_cfo[first_peak:first_peak+N*3+num_symbols]
rx_preamble = rx_frame[:N*2]
rx_data = rx_frame[N*3:]

# channel correction
rx_data_ch_cor = rx_data / np.mean(rx_preamble[:4])

plt.figure()
plt.plot(rx_data_ch_cor.real, rx_data_ch_cor.imag, '.')
plt.xlabel('I')
plt.ylabel('Q')
plt.title(r'IQ Constellation (M={})'.format(M))
plt.grid(True)
ax = plt.gca()
ax.set_aspect('equal', adjustable='datalim')
plt.savefig(r"IQ_Constellation_M{}.png".format(M))

# Demodulate symbols and compute BER
if M == 2 or M == 4:
    max_symbol_val = 1
elif M == 16:
    max_symbol_val = 3

demod_bits = np.zeros((rx_data_ch_cor.size, bits_per_symbol), dtype=np.int32)
for sym in range(rx_data_ch_cor.size):
    symbol = rx_data_ch_cor[sym] * mqamNormFactor
    real = np.clip(np.round((symbol.real + 1) / 2, 0) * 2 - 1, -max_symbol_val, max_symbol_val)
    if M == 2:
        symbol = real
    else:
        imag = np.clip(np.round((symbol.imag + 1) / 2, 0) * 2 - 1, -max_symbol_val, max_symbol_val)
        symbol = real + 1j * imag
    demod_bits[sym] = [int(bit) for bit in reversed(MEncodingRev[M][symbol])]

demod_bits = demod_bits.flatten()

print("Tx Bits", np.int32(bits))
print("Rx Bits", np.int32(demod_bits))

error_bits = np.zeros(min(len(demod_bits), len(bits)))
error_bits = demod_bits[:error_bits.shape[0]] - bits[:error_bits.shape[0]]
print("BER:", np.sum(np.abs(error_bits)) / error_bits.shape[0])