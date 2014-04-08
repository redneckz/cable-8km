
# coding: utf-8

## Формирующий фильтр

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

numpy.set_printoptions(threshold='nan')


## Введение

# *Формирующий фильтр* полезен при модуляции для подготовки сигнала к искажениям в канале. Спектр модулированного сигнала становится более компактным и менее подверженным МСИ. В случае QAM для этих целей используется *фильтр с косинусоидальным сглаживанием АЧХ* (cosflt).
# 
# ![АЧХ фильтра с косинусоидальным сглаживанием](images/sh.png)
# 
# ![Укрупненная структурная схема системы передачи информации](images/2698c04d.gif)
# 
# На практике прием и декодирование, как правило осуществляется при помощи согласованного фильтра G(f). Предположим, что модулятор и передатчик, а также приемник и демодулятор идеальные, т.е. сигнал на входе согласованного фильтра G(f) равен сигналу на выходе формирующего фильтра X(f) плюс аддитивный белый гауссов шум. Тогда общая частотная характеристика H(f) равна произведению H(f)=G(f)X(f). Для исключения МСИ необходимо чтобы H(f) соответствовала *cosflt*. При этом можно заметить, что G(f) должен быть согласован с исходным сигналом на выходе формирующего фильтра X(f), что означает - комплексное сопряженнае с формирующим фильтром. Тогда можно сказать, что |X(f)|<sup>2</sup>=H(f), а X(f)=sqrt(H(f)) *rcosflt*.

## Реализация scale/unscale на базе rcosflt

# In[2]:

FILTER_HN_TO_SCALE = 3

"""
Дельта-функция. Все кроме центрального отсчёта - нули.
@sample - исходный отсчёт
@dt - относительный номер отсчёта окрестности (относительно отсчёта @sample)
"""
def delta(sample, dt):
    return sample if (dt == 0) else 0.0

"""
Формирующий фильтр rcosflt 
@signal - исходный сигнал
@scale - кол-во отсчётов отфильтрованного сигнала на один отсчёт исходного
@betta - коэффициент сглаживания rcosflt; от 0 до 1; 0 - идеальный ФНЧ; 1 - без плоского участка в полосе пропускания
return - отмасштабированный сигнал
"""
def rcosflt_scale_signal(signal, scale, betta=0.7):
    if ((signal is None) or (len(signal) == 0)):
        return numpy.array([])
    
    # Порядок фильтра
    hn = FILTER_HN_TO_SCALE*scale
    # Импульсная характеристика sqrt фильтра с косинусоидальным сглаживанием
    h = rcosflt_h(scale, hn, betta)
    
    # Масштабирование
    scaled_signal = scale_signal(signal, scale, interpolation=lambda sample, dt: delta(sample, dt))
    
    # Компенсация фазы
    phase_shift = round(hn)
    # Отраженная голова
    signal_head = head(scaled_signal, phase_shift)
    # Отраженный хвост
    signal_tail = tail(scaled_signal, phase_shift)
    # Фильтрация
    filtered_signal = scipy.signal.lfilter(h, [1.0], numpy.concatenate([signal_head, scaled_signal, signal_tail]))
    return filtered_signal[2*phase_shift:]

"""
Возвращает центральный отсчёт
"""
def mid(samples, shift=0):
    if (samples is None): return None
    size = len(samples)-shift
    half_size = size//2
    return samples[shift+half_size]

"""
Согласованный к rcosflt фильтр. Обратная к @rcosflt_scale_signal функция
@signal - отмасштабированный сигнал
@scale - кол-во отсчётов отмасштабированного сигнала на один отсчёт исходного
@betta - коэффициент сглаживания rcosflt; от 0 до 1; 0 - идеальный ФНЧ; 1 - без плоского участка в полосе пропускания
return - исходный сигнал
"""
def rcosflt_unscale_signal(signal, scale, betta=0.7):
    # Порядок фильтра
    hn = FILTER_HN_TO_SCALE*scale
    # Импульсная характеристика sqrt фильтра с косинусоидальным сглаживанием
    h = rcosflt_h(scale, hn, betta)
    
    # Компенсация фазы
    phase_shift = round(hn)
    # Отраженная голова
    signal_head = head(signal, phase_shift)
    # Отраженный хвост
    signal_tail = tail(signal, phase_shift)
    # Фильтрация
    filtered_signal = scipy.signal.lfilter(h, [1.0], numpy.concatenate([signal_head, signal, signal_tail]))
    # Прореживание - центральные отсчёты каждого символа
    return unscale_signal(filtered_signal[2*phase_shift:], scale, reduction=mid)

"""
Отраженный относительно первого отсчёта кусок сигнала (голова/префикс) для целей фильтрации
@signal - сигнал
@size - длина отраженного куска
"""
def head(signal, size):
    res = numpy.empty(size)
    res.fill(signal[0])
    return res

"""
Отраженный относительно последнего отсчёта кусок сигнала (хвост/постфикс) для целей фильтрации.
@signal - сигнал
@size - длина отраженного куска
"""
def tail(signal, size, odd=True):
    res = numpy.empty(size)
    res.fill(signal[-1])
    return res

"""
Импульсная характеристика sqrt фильтра с косинусоидальным сглаживанием (rcosflt)
@scale - кол-во отсчётов отфильтрованного сигнала на один отсчёт исходного
@n - длина характеристики
@betta - коэффициент сглаживания rcosflt; от 0 до 1; 0 - идеальный ФНЧ; 1 - без плоского участка в полосе пропускания
"""
def rcosflt_h(scale, n, betta=0.7):
    # 2n+1 отсчётов характеристики -n..0..n
    indices = numpy.array(xrange(2*n + 1)) - n
    h = numpy.array(map(lambda index: rcosflt_ht(index, scale, betta), indices))
    return h/numpy.linalg.norm(h)


"""
Значение импульсной характеристики rcosflt в точке @t
@t - время
@ts - длительность/период символа
@betta - коэффициент сглаживания rcosflt; от 0 до 1; 0 - идеальный ФНЧ; 1 - без плоского участка в полосе пропускания
"""
def rcosflt_ht(t, ts, betta=0.7, eps=1e-5):
    fs = 1.0/ts
    wst = pi*fs*t
    four_betta = 4.0*betta
    if (abs(t) < eps):
        return sqrt(fs)*(1.0 - betta + four_betta/pi)
    if (abs(abs(t) - ts/four_betta) < eps):
        return (betta/sqrt(2.0*ts))*((1.0 + 2.0/pi)*sin(pi/four_betta) + (1.0 - 2.0/pi)*cos(pi/four_betta))
    a = sqrt(fs)*(sin((1.0-betta)*wst) + four_betta*fs*t*cos((1.0+betta)*wst))
    b = wst*(1.0 - (four_betta*fs*t)**2)
    return a/b


# http://en.wikipedia.org/wiki/Root-raised-cosine_filter

## Поясняющие графики

# In[3]:

SCALE = 32
N = 3*SCALE
BETTA = 0.7

b = rcosflt_h(SCALE, N, BETTA)
w, h = signal.freqz(b)

fig = pyplot.figure()
pyplot.title("rcosflt frequency response")
ax1 = fig.add_subplot(111)

pyplot.plot(w[:N//2], abs(h)[:N//2], "b")
pyplot.ylabel("Amplitude", color="b")
pyplot.xlabel("Frequency [rad/sample]")

ax2 = ax1.twinx()
pyplot.plot(w[:N//2], numpy.unwrap(numpy.angle(h))[:N//2], "g")
pyplot.ylabel("Angle (radians)", color="g")
pyplot.grid()
pyplot.axis("tight")
pyplot.show()

test_signal = 5.0-1.0*numpy.random.random_integers(10, size=32)
pyplot.plot(xrange(len(test_signal)), test_signal, "r")
pyplot.title("Test")
pyplot.grid()

pyplot.show()

scaled_signal = rcosflt_scale_signal(test_signal, SCALE, betta=BETTA)

pyplot.plot(xrange(len(scaled_signal)), scaled_signal, "r")
pyplot.title("Scaled")
pyplot.grid()

pyplot.show()

unscaled_signal = rcosflt_unscale_signal(scaled_signal, SCALE, betta=BETTA)

pyplot.plot(xrange(len(unscaled_signal)), unscaled_signal, "r")
pyplot.title("Unscaled")
pyplot.grid()

pyplot.show()

diff = numpy.abs(unscaled_signal-test_signal)

pyplot.plot(xrange(len(diff)), diff, "r")
pyplot.title("Diff")
pyplot.grid()

pyplot.show()

