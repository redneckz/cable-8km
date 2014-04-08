
# coding: utf-8

## Функции для построения цифровых фильтров

# Данный документ/модуль содержит функции построяения цифровых фильтров и используется в моделе кабеля/среды.

## Зависимости

# In[1]:

# Научные модули Python
import math
from math import *
import numpy
import scipy
from scipy import signal

# Модуль для построения графиков "как в Matlab"
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as pyplot

# Модуль общих функций
from common import *


## Цифровой фильтр без ОС

# In[2]:

"""
Функция фильтраци. По сути это функция свёртки сигнала с импульсной характеристикой фильтра
@signal - исходный сигнал
@h - импульсная характеристика фильтра
@phase - величина корректировки фазы после фильтрации 
@prev_signal - сигнал "присоединяемый" к началу исходного, по умолчанию это нулевой сигнал;
               в случае гармонического сигнала - это может быть сам сигнал
return - отфильтрованный сигнал
"""
def filter_signal(signal, h, phase=0, prev_signal=None):
    if ((signal is None) or (len(signal) == 0) or (h is None) or (len(h) == 0)):
        return numpy.array([])
    def xh(n):
        return filter_signal_n(signal, h, n, prev_signal)
    filtered_signal = numpy.array(map(xh, xrange(len(signal) + len(h))))
    return filtered_signal[phase:phase+len(signal)]

"""
Вычисляет один отсчёт отфильтрованного сигнала.
@signal - исходный сигнал
@h - импульсная характеристика фильтра
@n - номер отсчёта
@prev_signal - сигнал "присоединяемый" к началу исходного, по умолчанию это нулевой сигнал;
               в случае гармонического сигнала - это может быть сам сигнал
return - отсчёт отфильтрованного сигнала
"""
def filter_signal_n(signal, h, n, prev_signal=None):
    if ((signal is None) or (len(signal) == 0) or (h is None) or (len(h) == 0)):
        return numpy.array([])

    def x(i):
        if prev_signal is None:
            if ((i >= 0) and (i < len(signal))):
                return signal[i]
            elif (i < 0):
                return 0.0
            else:
                return signal[-1]
        else:
            if ((i >= 0) and (i < len(signal))):
                return signal[i]
            elif (i < 0):
                return prev_signal[len(prev_signal) + i]
            else:
                return signal[-1]

    return sum([x(n-len(h) + k+1)*h[k] for k in xrange(len(h))])

"""
Детектирует шаблон в сигнале (от начала)
@signal - сигнал
@tmpl - шаблон
@k - затухание
return - смещение в @signal максимально коррелирующее с @tmpl
"""
def detect_tmpl(signal, tmpl, k=1.0):
    if ((signal is None) or (len(signal) == 0) or (tmpl is None) or (len(tmpl) == 0)):
        return None
    """
    Функция скалярного умножения сигнала на шаблон (функция свёртки)
    @shift - смещение сигнала
    return - возвращает степень сходства смещенного сигнала/буфера с шаблоном
    """
    def similarity(shift):
        signal_last_index = min(len(signal), shift+len(tmpl))
        return sum(tmpl[0:signal_last_index-shift]*signal[shift:signal_last_index])/(signal_last_index-shift)
    
    # Всевозможные смещения шаблона относительно сигнала
    available_shifts = xrange(len(signal))
    # Мощность шаблона
    tmpl_pwr = sum(tmpl**2)
    # Минимально допустимое сходство.
    min_similarity = (0.5*tmpl_pwr/k)/len(tmpl)
    
    best_shift = None
    
    if (len(available_shifts) > 2):
        possible_shifts = []
        # Значения степени сходства для всевозможных смещений
        similarities = map(similarity, available_shifts)
        # Первая производная возможных степеней сходства
        d_similarities = numpy.diff(similarities)
        # Цикл поиска локальных максимумов (проивзодная "переходит" через 0)
        for i in xrange(1, len(d_similarities)-1):
            if ((d_similarities[i-1] > 0) and (d_similarities[i] < 0) and (similarities[i] > min_similarity)):
                best_shift = available_shifts[i]
                break
    else:
        possible_shift = max(available_shifts, key=similarity)
        if (similarity(possible_shift) > min_similarity):
            best_shift = possible_shift

    return best_shift


## НЧ фильтр

# Импульсная характеристика идеального НЧ фильтра - это *sinc* функция. Спектр *sinc*: H = 1, при f < fc; H = 0, при f >= fc
# Так как функция имеет бесконечную мощность, на практике применяется её "урезанная" версия, что приводит к искажению её спектра (пилообразные составляющие). Для сглаживания спектра используется *оконная функция Хемминга*. То есть импульсная характеристика НЧ фильтра на практике - это *sinc * hamming*

# In[3]:

"""
Импульсная характеристика НЧ фильтра
@critical_freq - частота среза НЧ фильтра
@sampling_freq - частота дискретизации
return - импульсная характеристика
"""
def low_pass_filter_h(critical_freq, sampling_freq):
    period_count = 7.5 # Кол-во периодов sinc до "удовлетворительного" затухания
    dt = 1.0/sampling_freq
    hn = int(round(period_count*sampling_freq/critical_freq)) # Кол-во целых отсчётов в 7.5 периодах
    t = numpy.linspace(-hn*dt, hn*dt, 2*hn)
    h = numpy.sinc(2*critical_freq*t) * hamming_window(len(t))
    return h/numpy.linalg.norm(h) # Нормировка

"""
Оконная функция Хемминга
@n - кол-во отсчётов
"""
def hamming_window(n):
    return (0.54 - 0.46*numpy.cos(2*pi*numpy.arange(n)/n))


## Поясняющие графики

# In[4]:

# Функция Хемминга
example_hamming = hamming_window(128)

# График функции Хемминга
pyplot.plot(xrange(len(example_hamming)), example_hamming, "r")
pyplot.title("Hamming Window")
pyplot.grid()

pyplot.show()

example_fd = 1000 # Частота дискретизации 1КГц
example_td = 1.0/example_fd
# Импульсная характеристика НЧ фильтра (частота среза 100Гц)
example_h = low_pass_filter_h(100, example_fd)

# График импульсной характеристики НЧ фильтра
pyplot.plot(numpy.arange(len(example_h))*example_td, example_h, "r")
pyplot.title("Low-pass filter h")
pyplot.xlabel("Time (sec)")
pyplot.grid()

pyplot.show()

# АЧХ импульсной характеристики НЧ фильтра
example_H = abs(numpy.fft.rfft(example_h))

# График АЧХ импульсной характеристики НЧ фильтра
pyplot.plot(numpy.fft.rfftfreq(len(example_h), example_td), example_H, "y")
pyplot.title("AFC")
pyplot.xlabel("Freq (Hz)")
pyplot.grid()

pyplot.show()


## Тесты

## Тестовый сигнал

# In[5]:

test_n = 256 # Кол-во отсчётов тестового сигнала
test_f = 100 # Наименьшая отличная от нуля частота тестового сигнала
test_period_count = 10.0 # Кол-во периодов тестового сигнала
test_t = numpy.linspace(0.0, test_period_count/test_f, test_n)
# Базовый сигнал sin(wt), подлежащий восстановлению фильтром
test_base_signal = numpy.sin(2*pi*test_f*test_t)
# Тестовый сигнал sin(wt) + 0.5sin(2wt) + 0.5sin(3wt)
test_signal = test_base_signal + 0.25*numpy.sin(2*pi*2*test_f*test_t) + 0.5*numpy.sin(2*pi*3*test_f*test_t)

# График тестового сигнала
pyplot.plot(test_t, test_signal, "r")
pyplot.title("Test signal")
center_plot_axis()
pyplot.xlabel("Time (sec)")
pyplot.grid()

pyplot.show()

# АЧХ тестового сигнала
test_signal_afc = 2*abs(numpy.fft.rfft(test_signal))/len(test_signal)

# График АЧХ тестового сигнала
test_fd = test_f*test_n/test_period_count # Частота дискретизации тестового сигнала
test_dt = 1.0/test_fd
pyplot.plot(numpy.fft.rfftfreq(len(test_signal), test_dt), test_signal_afc, "y")
pyplot.title("AFC")
pyplot.xlabel("Freq (Hz)")
pyplot.grid()

pyplot.show()


## НЧ Фильтр и тестовый сигнал после фильтрации

# In[16]:

b = signal.firwin(64, 1.7*test_f, nyq=test_fd/2.0)
w, h = signal.freqz(b)

fig = pyplot.figure()
pyplot.title("Digital filter frequency response")
ax1 = fig.add_subplot(111)

pyplot.plot(w, abs(h), "b")
pyplot.ylabel("Amplitude [dB]", color="b")
pyplot.xlabel("Frequency [rad/sample]")

ax2 = ax1.twinx()
pyplot.plot(w, numpy.unwrap(numpy.angle(h)), "g")
pyplot.ylabel("Angle (radians)", color="g")
pyplot.grid()
pyplot.axis("tight")
pyplot.show()

# Отфильтрованный тестовый сигнал
filtered_test_signal = signal.filtfilt(b, [1.0], test_signal)
#filtered_test_signal = filter_signal(test_signal, low_pass_test_h, phase=len(low_pass_test_h)//2)

# График отфильтрованного тестового сигнала
pyplot.plot(test_t[0:len(filtered_test_signal[20:])], filtered_test_signal[20:], "r")
pyplot.title("Test signal after low-pass filter")
pyplot.ylim(-1.1, 1.1)
center_plot_axis()
pyplot.xlabel("Time (sec)")
pyplot.grid()

pyplot.show()

# АЧХ отфильтрованного тестового сигнала
filtered_test_signal_afc = 2*abs(numpy.fft.rfft(filtered_test_signal))/len(filtered_test_signal)

# График АЧХ отфильтрованного тестового сигнала
pyplot.plot(numpy.fft.rfftfreq(len(filtered_test_signal), test_dt), filtered_test_signal_afc, "y")
pyplot.title("AFC")
pyplot.xlabel("Freq (Hz)")
pyplot.grid()

pyplot.show()

# График разницы между базовым сигналом и отфильтрованным тестовым сигнала
pyplot.plot(test_t[0:len(filtered_test_signal)], test_base_signal[0:len(filtered_test_signal)] - filtered_test_signal, "r")
pyplot.title("Diff between Base and Filtered signals")
center_plot_axis()
pyplot.xlabel("Time (sec)")
pyplot.grid()

pyplot.show()

