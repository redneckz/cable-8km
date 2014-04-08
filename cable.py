
# coding: utf-8

## Упрощенная модель кабеля/среды

# Данный документ/модуль содержит упрощенную *модель кабеля/среды*. Которая состоит из *НЧ фильтра* и генератора белого шума.
# 
# Более точная модель не требуется в силу итерационного характера проекта; теоретическая и практическая части перемежаются и реализуются, по возморжности, одновременно.
# 
# Для иллюстрации далее приводится абстрактная модель кабеля (распределённый импеданс и утечка). 
# 
# ![Схема кабеля](images/image002_135.gif)
# 
# В данной моделе не учитывается индуктивная составляющая и распределённость импеданса (функция от расстояния).

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

# Модуль цифровой фильтрации
import digital_filter


## Функция кабеля/среды

# In[2]:

def cable_b(critical_freq, sampling_freq):
    return signal.firwin(128, cutoff=critical_freq, width=0.9*sampling_freq, nyq=sampling_freq/2)

"""
Данная функция моделирует кабель/среду, а именно: зашумление, затухание, и фильтрацию.
@signal - сигнал
@att - затухание (Дб)
@critical_freq - максимальная частота полосы пропускания кабеля (Гц)
@sampling_freq - частота дискретизации (Гц)
@noise_amp - амплитуда белого шума
@intersample_relative_offset - дополнительное смещение приёмника относительно источника в долях отсчёта
return - (список частот, список амплитуд)
"""
def cable(signal, att, critical_freq, sampling_freq, noise_amp=0.0, intersample_relative_offset=0.0):
    if ((signal is None) or (len(signal) == 0)): return numpy.array([])
    # Импульсная характеристика НЧ фильтра
    filter_b = cable_b(critical_freq, sampling_freq)
    # Отфильтрованный сигнал
    filtered_signal = scipy.signal.lfilter(filter_b, [1.0], signal)
    # Отфильтрованный и зашумлённый сигнал
    filtered_noised_signal = filtered_signal + noise(len(filtered_signal), noise_amp)
    # Дополнительное смещение
    filtered_noised_and_shifted_signal = []
    if (len(filtered_noised_signal) > 1):
        for i in xrange(0, len(filtered_noised_signal)-1):
            prev_sample = filtered_noised_signal[i]
            sample = filtered_noised_signal[i+1]
            filtered_noised_and_shifted_signal.append(intersample_relative_offset*prev_sample +
                                                      (1.0-intersample_relative_offset)*sample)
        # Последний отсчёт остаётся неизменным
        filtered_noised_and_shifted_signal.append(filtered_noised_signal[-1])
    else:
        filtered_noised_and_shifted_signal = filtered_noised_signal
    # Затухание
    k = 10**(att/10)
    return (1.0/k)*numpy.array(filtered_noised_and_shifted_signal)

"""
Белый шум
@n - кол-во отсчётов
@amp - амплитуда
return - шум
"""
def noise(n, amp):
    return amp*(2*numpy.random.random(n) - 1)

