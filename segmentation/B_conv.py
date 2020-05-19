import numpy as np
# from scipy.signal import argrelextrema, argrelmin, argrelmax
# from scipy import signal as SIG
# from scipy.signal import savgol_filter
import pandas as pd
from scipy.signal import medfilt
import sys

# def smooth(x,window_len=100,window='hanning'):
#     if x.ndim != 1:
#         raise ValueError ("smooth only accepts 1 dimension arrays.")
#     if x.size < window_len:
#         raise ValueError ("Input vector needs to be bigger than window size.")
#     if window_len<3:
#         return x
#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError ("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
#     s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#     #print(len(s))
#     if window == 'hanning': #moving average
#         w=np.ones(window_len,'d')
#     else:
#         w=eval('np.'+window+'(window_len)')
#     y=np.convolve(w/w.sum(),s,mode='valid')
#     return y

# def multi_smooth (x, n):
#     for i in range(n):
#         x = smooth(x)
#     return x


def mad_scaling (signal):
    shift = np.median(signal)
    scale = np.median(np.abs (signal - shift))
    return (signal - shift)/scale

def turning_points(array):
    idx_max, idx_min = [], []
    if (len(array) < 3):
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)
    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING:
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max



# head = True
with open(sys.argv[1], 'rt') as s:
    for l in s:
        # if head:
        #     head = False
        #     continue
        # print(l)
        l = l.strip('\n')
        l = l.split('\t')
        rd = l[0]
        sig = [int(i) for i in l[8:]]
        raw_signal = np.array (list (map (int, sig[0:20000])))
        signal = mad_scaling (raw_signal)
        x = mad_scaling (list (range (len(signal))))
        v = np.hstack([np.ones(signal.shape), np.ones(signal.shape) * - 1])
        conv = np.convolve(medfilt (signal,2001), v, mode='valid')
        y = list (conv/1000)
        valleys, peaks = turning_points(list(conv))
        # fig_out = rd+'.pdf'
        # f = plt.figure()
        # plt.title (rd)
        # plt.plot(signal, label = 'scaled squiggle')
        # plt.plot (y, label= 'conv')
        # plt.legend (loc='best')
        begin, end = '',''
        try:
            if peaks[0]>=1000:
                begin = valleys[0]
                end = peaks[0]
                # plt.axvline(x=begin,color='r')
                # plt.axvline(x=end,color='r')
            else:
                begin = valleys[0]
                end = peaks[1]
                # plt.axvline(x=begin,color='r')
                # plt.axvline(x=end,color='r')
            # f.savefig(fig_out)
            # adapter_sig = ",".join (sig[begin:end])
            # print (rd,begin,end,adapter_sig,sep='\t')
            print(rd,begin,end,sep='\t')
        except:
            continue
            # print (rd, file=sys.stderr)
            # print (peaks, file=sys.stderr)
            # print (valleys, file=sys.stderr)
        # plt.show()

'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import pandas as pd
import argparse
import time
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from itertools import groupby

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def mad_scaling (signal):
    shift = np.median(signal)
    scale = np.median(np.abs (signal - shift))
    sd = np.std(signal)
    return ((signal - shift)/scale)

def der (s,win,order):
    """
    inputs are:
    s: list or numpy array
    win: window size
    """
    return savgol_filter (s, window_length=win, polyorder=2, deriv = order)

def sig_conv(read, signal, der_size, seg_size):
	#print ('processing',read,file=sys.stderr)
	try:
		s = np.array (signal).astype(int)
		s = mad_scaling (s)
		v = np.hstack([np.ones(s.shape), np.ones(s.shape) * - 1])
		conv = np.convolve (s, v, mode = 'valid')
		y = list (conv/1000)
		#valleys, peaks = turning_points(list(conv2))
		der1 = der (conv, der_size, 1)
		sign = np.asarray (der1 > 0).astype (int)
		segments = np.array ([list(grp) for k, grp in groupby(sign)])
		#print (sign[:3])
		new_sign = []
		#new_sign.append (segments[0])
		for i in range (len(segments))[0:]:
			if len(segments[i]) <= seg_size and i >0:
				new_sign[-1] = new_sign[-1] + [new_sign[-1][0] for _ in segments[i]]
			else:
				new_sign.append (segments[i])
		new_segments = new_sign

		coors = []
		pos = 0

		if new_segments[0][0] == 1 and len (new_segments[0])<1000:
			pos += len(new_segments[0] + new_segments[1])
			coors.append(pos)
			for i in new_segments[2:]:
				if (len(coors)) > 3:
					break
				pos += len (i)
				coors.append(pos)
		elif  new_segments[0][0] == 1 and len (new_segments[0])<1000: #no leader
			print (read,'no leader', sep='\t',file=sys.stderr)
			coors.append(pos)
			for i in new_segments:
				if (len(coors)) > 3:
					break
				pos += len (i)
				coors.append(pos)
		else:
			for i in new_segments:
				if (len(coors)) > 3:
					break
				pos += len (i)
				coors.append(pos)

		cnt = 0
		colors = ['k','g','r','c','m']
		plt.plot(y)
		plt.plot(s)
		print (read, "\t".join (map (str, coors[0:2])), sep="\t", file=sys.stdout)
		for xk in coors[0:2]:
				#color = colors[cnt%2]
			plt.axvline(x=xk+10,color='k')
		pdf = read + '.pdf'
		plt.savefig (pdf)
		plt.close()
	except:
		raise
		print ('error occured when processing' ,read,  str(IOError) )

def main():
	der_size = int (sys.argv[2])
	seg_size = int (sys.argv[3])
	fh = open (sys.argv[1],'r')
	print ('#read\tbarcode_begin\tbarcode_end',file=sys.stdout)
	for l in fh:
		ary = l.rstrip().split()
		read, signal = ary[0], ary[1:20000]
		sig_conv(read, signal, der_size, seg_size)

if __name__ == '__main__':
	main()
'''
