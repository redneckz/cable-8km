
# coding: utf-8

## Модуль общих функций

## Зависимости

# In[20]:

# Научные модули Python
import math
import numpy

# Модуль для построения графиков "как в Matlab"
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as pyplot

# Модуль Python для работы с массивами бит
import bitarray


## Функции для работы с последовательностями бит

# In[21]:

"""
Преобразует строку в последовательность бит
@s - текст
return - последовательность бит, соответстсвующая тексту
"""
def string_to_bit_seq(s):
    ba = bitarray.bitarray()
    ba.fromstring(s)
    return numpy.array(map(lambda bit: 1 if bit else 0, ba))

"""
Преобразует число в последовательность бит
@num - целое число
@bit_count - разрядность
@least_significant_first - по умолчанию true, то есть младшие биты идут первыми
return - последовательность бит, соответстсвующая числу
"""
def number_to_bit_seq(num, bit_count, least_significant_first=True):
    integer = int(num)
    bit_seq = numpy.array([((integer>>i)&1) for i in xrange(bit_count)])
    return bit_seq if least_significant_first else bit_seq[::-1]

"""
Преобразует последовательность бит в число
@bit_seq - последовательность бит
return - целое число, соответстсвующее последовательности бит
"""
def bit_seq_to_number(bit_seq):
    return int(sum(map(lambda (i, x): x * 2**i, enumerate(bit_seq))))

"""
@num - целое число
return - число единичных бит в num
"""
def bit_count(num):
    count = 0
    while(num):
        if (num&1): count = count+1
        num = num>>1
    return count;
    
"""
@num - целое число
return - результат операции xor всех бит числа num между собой
"""
def xor_bits(num):
    return bit_count(num)%2


## Код Грея

# In[22]:

"""
Функция генерации кода Грея
@index - номер
return - код
"""
def gray_encode(index):
    return index^(index>>1)

"""
Обратная к @gray_encode функция
@gray - код
return - номер
"""
def gray_decode(gray):
    index = 0
    while(gray):
        index = index^gray
        gray = gray>>1
    return index


## Вспомогательные функции алгоритмов модуляции

# In[23]:

"""
Функция усреднения массива/списка/коллекции чисел
@array - массив чисел
return - среднее значение
"""
def avg(array):
    return sum(array)/float(len(array))

"""
Преобразует последовательность бит в уровень от -amp до amp
@bit_seq - последовательность бит
@amp - амплитуда модуляции
@num_encoder - вспомогательная функция преобразования номера уровня (по умолчанию - код Грея)
return - уровень от -amp до amp
"""
def bit_seq_to_level(bit_seq, amp, num_encoder=gray_decode):
    num = num_encoder(bit_seq_to_number(bit_seq))
    max_num = 2**len(bit_seq) - 1
    return amp*(2.0*num/max_num - 1.0)

"""
Обратная к @ref bit_seq_to_level функция
@level - уровень
@bit_count - кол-во бит в исходной битовой последовательности
@amp - амплитуда модуляции
@num_decoder - обратная к @bit_seq_to_level/@num_encoder функция (по умолчанию - код Грея)
return - битовая последовательность соответстсвующая уровню
"""
def level_to_bit_seq(level, bit_count, amp, num_decoder=gray_encode):
    levels = numpy.linspace(-amp, amp, 2**bit_count)
    closest_level_index = abs(levels - level).argmin()
    max_num = 2**bit_count - 1
    num = num_decoder(int(round((levels[closest_level_index]/amp + 1.0)*max_num/2.0)))
    return number_to_bit_seq(num, bit_count)

"""
@sample - исходный отсчёт
@dt - относительный номер отсчёта окрестности (относительно отсчёта @sample)
"""
def const(sample, dt):
    return sample

"""
Функция масштабирования вдоль оси времени, путём интерполяции
@signal - исходный сигнал
@scale - кол-во отсчётов отмасштабированного сигнала на один отсчёт исходного
@interpolation - функция интерполяции окрестности отсчёта; вычисояет значения ближайших
return - отмасштабированный сигнал
"""
def scale_signal(signal, scale, interpolation=const):
   return numpy.array([interpolation(signal[i//scale], i%scale - scale//2) for i in xrange(scale*len(signal))])

"""
Обратная к @ref scale_signal функция
@signal - отмасштабированный сигнал
@scale - кол-во отсчётов отмасштабированного сигнала на один отсчёт исходного
@reduction - функции свёртки последовательности из scale отсчётов отмасштабированного сигнала
             в один сигнал исходного; по умолчанию используется функция усреднения @ref avg
return - исходный сигнал
"""
def unscale_signal(signal, scale, reduction=avg):
    return numpy.array([reduction(signal[i*scale:(i+1)*scale]) for i in xrange(len(signal)//scale)])

"""
Вычисляет АЧХ сигнала
@signal - сигнал
@sampling_freq - частота дискретизации
return - (список частот, список амплитуд)
"""
def compute_signal_afc(signal, sampling_freq):
    sampling_period = 1.0/sampling_freq
    t = numpy.linspace(0, len(signal)*sampling_period, len(signal))
    freq = numpy.fft.rfftfreq(len(t), 1.0/sampling_freq)
    signal_fft = numpy.fft.rfft(signal)
    afc = 2*abs(signal_fft)/len(signal)
    return (freq, afc)

"""
Вычисляет АЧХ/ФЧХ сигнала
@signal - сигнал
@sampling_freq - частота дискретизации
return - (список частот, список амплитуд, список фаз)
"""
def compute_signal_afc_pfc(signal, sampling_freq):
    sampling_period = 1.0/sampling_freq
    t = numpy.linspace(0, len(signal)*sampling_period, len(signal))
    freq = numpy.fft.rfftfreq(len(t), 1.0/sampling_freq)
    signal_fft = numpy.fft.rfft(signal)
    afc = 2*abs(signal_fft)/len(signal)
    pfc = numpy.angle(signal_fft)
    return (freq, afc, pfc)


## Вспомогательные функции построения графиков

# In[24]:

"""
Центрует точку отсчёта при построении графиков
"""
def center_plot_axis(center=(0, 0)):
    gca = pyplot.gca()
    gca.spines["right"].set_position(("data", center[0]))
    gca.spines["top"].set_position(("data", center[1]))

