from librosa import load, stft, istft
import soundfile as sf
import numpy as np

import argparse

def phase_vocoder(stft_matrix, r):

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # эта дает перекрытие в 75%, как в рекомендуется в алгоритме
    hop_length = int(n_fft // 4)

    time_frames = np.arange(0, stft_matrix.shape[1], r)

    # создаем пустую матрицу, куда будем складывать ответ
    d_stretch = np.zeros((stft_matrix.shape[0], len(time_frames))) 

    # ожидаемое опережение по фазе на каждом шаге
    phi_advance = np.linspace(0, np.pi * hop_length, stft_matrix.shape[0]) 

    # берем угол поворота комплекной сотавляющей от вещественной части,
    # это даст фазу
    # будем суммировать эту фазу
    phase_sum = np.angle(stft_matrix[:, 0]) 

    for (i, step) in enumerate(time_frames):

        columns = stft_matrix[:, int(step) : int(step + 2)]

        # Считаем амплитуду 
        # модуль возвращает амплитуду
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns[:, 0]) + alpha * np.abs(columns[:, -1])

        d_stretch[:, i] = mag * np.exp(1.0j * phase_sum)

        # высчитываем опережение по фазе
        delta_phase = np.angle(columns[:, -1]) - np.angle(columns[:, 0]) - phi_advance

        # помещаем в интервал от -пи до пи
        delta_phase = delta_phase - 2.0 * np.pi * np.round(delta_phase / (2.0 * np.pi))

        # суммируем опережение по фазе
        phase_sum += phi_advance + delta_phase

    return d_stretch

def time_streching(y, r):

    # Будем выполнять все операции матрично

    #####  ANALYSIS  #####

    # Применяем к сигналу преобразование Фурье
    stft_matrix = stft(y)

    #####  PREPROCESSING  #####

    # Алгоритм phase vocoder, который расстягивает/сжимает фреймы без изменения питча
    stft_stretched = phase_vocoder(stft_matrix, r)

    # Посчитаем какой длины будет выход в зависимости от коэффициента r
    size_stretched = int(round(len(y) / r))

    #####  SYNTHESIS  #####

    # Применяем обратное преобразование Фурье для получения вейв формы
    y_stretched = istft(stft_stretched, length=size_stretched)

    return y_stretched

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run args.')
    parser.add_argument('--input', type=str, default='', help='path to input file')
    parser.add_argument('--output', type=str, default='', help='path to output file')
    parser.add_argument('--ratio', type=float, default=1., help='time stretch ratio')

    args = parser.parse_args()

    aud, sample_rate = load(args.input)
    stretched = time_streching(aud, args.ratio)

    sf.write(args.output, stretched, sample_rate)
