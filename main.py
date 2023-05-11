import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.fft import fft, fftfreq
import pandas as pd
import binascii

# -------------------=FUNÇÕES=-------------------


# criação de seno
def cont_sin(time, sample_rate, frequency):
    time = time
    sample = sample_rate
    f = frequency
    t = np.linspace(time, time + 0.1, sample)
    signal = np.sin(2 * np.pi * f * t)
    return t, signal


# -------------------MODULAÇÃO-------------------

# padrões de frequencia
Fs = 10000
T = 0
fc1 = 1300  # 1
fc2 = 1700  # 0

# resgata a mensagem
ascii_message = input("Insira a mensagem desejada: ")

# convere para binário
message = bin(int.from_bytes(ascii_message.encode(), "big"))
print("Input ASCII: " + ascii_message)
print("Input Bin: " + message)

# Converte a mensagem binária para uma sequência de frequências.
bin_message = "0" + message[2:]
freq_bin_message = np.zeros(len(bin_message))
for i in range(len(bin_message)):
    if bin_message[i] == "0":
        freq_bin_message[i] = fc2
    else:
        freq_bin_message[i] = fc1

# cria o sinal modulado
signal = np.zeros(0)
t = np.zeros(0)
for i in range(len(freq_bin_message)):
    time, bit_signal = cont_sin(T, Fs, freq_bin_message[i])
    signal = np.hstack([signal, bit_signal])
    t = np.hstack([t, time])
    T += 0.1

# exibe o gráfico
plt.plot(t, signal)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title(r"Senoidal")
plt.xlim([0, 0.001])
plt.show()

# espectro do sinal
T = 0.001  # calcular o período do sinal 0.001 -> 1/T = 1000
N = signal.size

# aplica o fourier para converter o sinal temporal em sinal de frequência
f = fftfreq(len(signal), T)
frequencias = f[: N // 2]
amplitudes = np.abs(fft(signal))[: N // 2] * 1 / N

print(
    "Value in index ",
    np.argmax(amplitudes),
    " is %.2f" % amplitudes[np.argmax(amplitudes)],
)
print("Freq: ", frequencias[np.argmax(amplitudes)])
plt.plot(frequencias, amplitudes)
plt.grid()
plt.xlim([0, 2000])
plt.show()

# ----------------------- Demodulador -----------------------

string_demodulada = "0b"  # usar para converter string_demodulada após a conversão de frequências em 0 ou 1
samples_bit = 10000

for bit_position in range(len(freq_bin_message)):
    # capturar uma quantidade de valores do sinal (vetor) dentro do tempo de bit adequado
    signal_result = signal[
        bit_position * samples_bit : (bit_position + 1) * samples_bit
    ]

    # T = 1 / Fs  # calcular o perído de amostragem (1/F sendo F 10000 Hz)
    N = signal_result.size  # pegar a quantidade de amostras do bit
    # aplicar FFT para saber qual é a frequência
    # f = fftfreq(len(signal_result), T)
    # frequencias = f[: N // 2]
    amplitudes = np.abs(fft(signal_result))[: N // 2] * 1 / N

    # parte onde é verificado se o bit (com uma quantidade de amostras que são usadas para detectar se é 0 ou 1) analisa se é 0 ou 1

    # Verifica se a é frequência de 1700Hz
    if freq_bin_message[bit_position] == fc2:
        if amplitudes[np.argmax(amplitudes)] > 0.5:
            string_demodulada += "1"
        else:
            string_demodulada += "0"
    # Caso contrário, atribui o valor de 1300Hz
    else:
        if amplitudes[np.argmax(amplitudes)] > 0.5:
            string_demodulada += "0"
        else:
            string_demodulada += "1"


# Converter Mensagem
print("Output Bin: " + string_demodulada)
n = int(string_demodulada, 2)
print("Output ASCII: " + binascii.unhexlify("%x" % n).decode("utf-8"))
