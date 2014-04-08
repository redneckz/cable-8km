
# coding: utf-8

## Синхронизация и детектирование начала пакета

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

# Модуль формирующего фильтра
from special_filter import *

numpy.set_printoptions(threshold='nan')

EPS = 0.000001


## Функция генерации несущей

# In[2]:

"""
Функция генерации несущей с учетом межсимвольного интервала
@w - циклическая частота
@symbol_count - кол-во символов
@symbol_period - длительность символа
@scale - кол-во отсчётов на символ
@phase - фаза
return - пару (отсчёты времени, отсчёты несущей)
"""
def carrier(w, symbol_count, symbol_period, scale, phase=0.0):
    t = numpy.linspace(0, symbol_count*symbol_period, symbol_count*scale, endpoint=False)
    return (t, numpy.sin(w*t + phase))


## Синхронизация по методу последовательных приближений (МПП/MSA)

# Данный метод точной корректировки фазы опирается на знание о характере сигнала (шаблон).
# * Шаблон должен располагаться в начале пакета.
# * Синфазная часть шаблона **I** должна состоять из отсчётов с максимальным уровнем.
# * Квадратурная часть шаблона **Q** должна быть противоположна синфазной **Q** = **-I**

# ![Синхронизация](images/synch.jpg)

# In[3]:

"""
Так как определение величины корректировки фазы происходит по МПП,
необходима малая величина, задающая точность корректировки (условие остановки поиска решения)
"""
MSA_SYNCH_EPS = pi/(16*180)

"""
Функция определения величины корректировки фазы по МПП
@mod_signal - модулированный сигнал, требующий синхронизации
@tmpl_symbol_count - кол-во символов в шаблоне
@max_mod_level - абсолютное значение уровня в каждом из каналов шаблона (максимальная амплитуда модуляции)
@w - циклическая частота несушей
@symbol_period - длительность символа
@scale - кол-во отсчётов на символ
@phase_shift - "нулевая" фаза
@eps - точность определния
@v - скорость поиска корректировки
@log - флаг включения логирования
return - пару (величина корректировки фазы с заданной точностью, коэффициент усиления)
"""
def msa_synch(mod_signal, tmpl_symbol_count, max_mod_level, w, symbol_period, scale,
              phase_shift=0.0, eps=MSA_SYNCH_EPS, v=1.0, log=False):
    # Значение корректировки фазы предыдущего шага
    prev_optimal_demod_phase = 0.0
    # Текущее значение корректировки фазы и коэфф. усиления
    optimal_demod_phase, gain = msa_synch_step(mod_signal, tmpl_symbol_count, max_mod_level, w, symbol_period, scale,
                                               phase=phase_shift)
    while (abs(optimal_demod_phase-prev_optimal_demod_phase) > eps):
        if log: print "* {}рад".format(optimal_demod_phase)
        prev_optimal_demod_phase = optimal_demod_phase
        new_delta_phase, new_gain = msa_synch_step(mod_signal, tmpl_symbol_count, max_mod_level, w, symbol_period, scale,
                                           phase=phase_shift + optimal_demod_phase)
        if (not(new_delta_phase is None)): delta_phase, gain = new_delta_phase, new_gain
        optimal_demod_phase = optimal_demod_phase + v*delta_phase
    return (optimal_demod_phase, gain)

"""
Функция одной итерации корректировки фазы по МПП
@mod_signal - модулированный сигнал, требующий синхронизации
@tmpl_symbol_count - кол-во символов в шаблоне
@max_mod_level - абсолютное значение уровня в каждом из каналов шаблона (максимальная амплитуда модуляции)
@w - циклическая частота несушей
@symbol_period - длительность символа
@scale - кол-во отсчётов на символ
@phase - текущая фаза
return - пару (корректировка фазы, коэффициент усиления)
"""
def msa_synch_step(mod_signal, tmpl_symbol_count, max_mod_level, w, symbol_period, scale, phase=0.0):
    carrier_cos = carrier(w, tmpl_symbol_count, symbol_period, scale, math.pi/2.0 + phase)
    carrier_sin = carrier(w, tmpl_symbol_count, symbol_period, scale, phase)
    # Демодуляция шаблона
    mod_tmpl = mod_signal[:tmpl_symbol_count*scale]
    demod_tmpl_synch_part = rcosflt_unscale_signal(2*mod_tmpl*carrier_cos[1], scale)
    demod_tmpl_quad_part = rcosflt_unscale_signal(2*mod_tmpl*carrier_sin[1], scale)
    # Мощность демодулированного шаблона
    demod_tmpl_pwr = sum(demod_tmpl_synch_part**2)+sum(demod_tmpl_quad_part**2)
    if (demod_tmpl_pwr < EPS): return (None, None)
    # Затухание в канале
    k = sqrt(demod_tmpl_pwr/(2*tmpl_symbol_count*(max_mod_level**2)))
    # Коэфф. усиления
    gain = 1.0/k
    # Приблизительная величина корректировки фазы
    delta_phase = asin(gain*(sum(demod_tmpl_synch_part)+sum(demod_tmpl_quad_part))/(2*tmpl_symbol_count*max_mod_level))
    return (delta_phase, gain)


## Синхронизация по методу Feedforward Carrier Recovery (FCR)

# Данный метод основан на переборе (ненаправленный поиск) с минимизацией среднеквадратичной ошибки. Таким образом, точность сильно зависит от диапазона и шага перебора. Может использоваться как для поиска начала пакета, так и для точной непрерывной подстройки фазы без опоры на шаблон.
# 
# Более подробная информация [здесь](docs/Phase-Noise-Tolerant Carrier recovery .pdf).
# 
# При поиске начала пакета по этому методу характер шаблона не играет роли.

# In[4]:

FCR_TH = 0.1

"""
Детектирование пакета по методу FCR с шаблоном
@mod_signal - модулированный сигнал
@max_offset - задаёт диапазон поиска @mod_signal[0:@max_offset]
@tmpl_synch_part - уровни синфазной части шаблона
@test_symbol_count - кол-во символов (включая символы шаблона) учавствующих в детектировании
@w - циклическая частота несушей
@symbol_period - длительность символа
@scale - кол-во отсчётов на символ
@closest_level - функция определения ближайшего уровня
@th - пороговое значение среднеквадратичной ошибки
return - пару (номер отсчёта в @mod_signal, коэффициент усиления)
"""
def fcr_detect(mod_signal, max_offset, tmpl_synch_part, test_symbol_count,
               w, symbol_period, scale, closest_level, th=FCR_TH):
    carrier_cos = carrier(w, test_symbol_count, symbol_period, scale, math.pi/2.0)
    carrier_sin = carrier(w, test_symbol_count, symbol_period, scale)
    
    """
    Оценка ошибки для заданного смещения в @mod_signal
    return - пару (величина ошибки, коэффициент усиления)
    """
    def estimate_error(offset):
        # Демодуляция тестовой последовательности символов
        test = mod_signal[offset:offset+(test_symbol_count*scale)]
        test_synch_part = rcosflt_unscale_signal(2*test*carrier_cos[1], scale)
        test_quad_part = rcosflt_unscale_signal(2*test*carrier_sin[1], scale)
        # Кол-во символов в шаблоне
        tmpl_symbol_count = len(tmpl_synch_part)
        # Мощность демодулированного шаблона
        demod_tmpl_pwr = sum(test_synch_part[:tmpl_symbol_count]**2)+sum(test_quad_part[:tmpl_symbol_count]**2)
        if (demod_tmpl_pwr < EPS): return (float("inf"), float("inf"))
        # Эталонная мощность
        tmpl_pwr = 2*sum(tmpl_synch_part**2)
        # Затухание в канале
        k = sqrt(demod_tmpl_pwr/tmpl_pwr)
        # Коэфф. усиления
        gain = 1.0/k
        # Усиление
        amplified_synch_part = gain*test_synch_part
        amplified_quad_part = gain*test_quad_part
        # Величина ошибки
        e = fcr_error(amplified_synch_part, closest_level, tmpl=tmpl_synch_part) + fcr_error(
                      amplified_quad_part, closest_level, tmpl=-tmpl_synch_part)
        # Величина ошибки и коэфф. усиления
        return (e, gain)
    
    errors = map(estimate_error, xrange(max_offset))
    
    pyplot.plot(xrange(len(errors)), errors, "r")
    pyplot.title("FCR Detect Error")
    pyplot.xlabel("Index")
    pyplot.grid()
    pyplot.show()
    
    # Смещение с наименьшей ошибкой
    best_offset = numpy.argmin(errors, axis=0)[0]
    # Наименьшая ошибка и соответствующий коэфф. усиления
    min_error = errors[best_offset]
    # Если наименьшая ошибка меньше порогового значения,
    if (min_error[0] < th):
        # то возвращаем наилучшее смещение и соответствующий коэфф. усиления
        return (best_offset, min_error[1])
    else:
        return (None, None) # Пакет не обнаружен

"""
Функция определения величины корректировки фазы по FCR. Может использоваться в непрерывном режиме,
так как нет опоры на шаблон (non-data-aided)
@mod_signal - модулированный сигнал
@phases - список фаз для поиска
@test_symbol_count - кол-во символов учавствующих в корректировке фазы
@w - циклическая частота несушей
@symbol_period - длительность символа
@scale - кол-во отсчётов на символ
@closest_level - функция определения ближайшего уровня
@gain - коэфф усиления
return - величина корректировки фазы
"""
def fcr_synch(mod_signal, phases, test_symbol_count, w, symbol_period, scale, closest_level, gain=1.0):
    """
    Оценка ошибки для данного значения фазы
    """
    def estimate_error(phase):
        carrier_cos = carrier(w, test_symbol_count, symbol_period, scale, math.pi/2.0 + phase)
        carrier_sin = carrier(w, test_symbol_count, symbol_period, scale, phase)
        # Демодуляция тестовой последовательности символов
        test = mod_signal[:test_symbol_count*scale]
        test_synch_part = rcosflt_unscale_signal(2*test*carrier_cos[1], scale)
        test_quad_part = rcosflt_unscale_signal(2*test*carrier_sin[1], scale)
        amplified_synch_part = gain*test_synch_part
        amplified_quad_part = gain*test_quad_part
        return fcr_error(amplified_synch_part, closest_level) + fcr_error(amplified_quad_part, closest_level)
    errors = map(estimate_error, phases)
    best_phase_index = numpy.argmin(errors)
    return phases[best_phase_index]

"""
Функция определения среднекваратичной ошибки для заданной последовательности уровней
@levels - последовательность уровней
@closest_level - функция определения ближайшего уровня
@tmpl - шаблон
return среднекваратическая ошибка отклонения от идеальных уровней
"""
def fcr_error(levels, closest_level, tmpl=[]):
    closest_levels = numpy.concatenate([tmpl, map(closest_level, levels[len(tmpl):])])
    return sum((closest_levels-levels)**2)/len(levels)

