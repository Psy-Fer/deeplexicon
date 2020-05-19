import numpy as np
import sys, os
import pandas as pd
import argparse
import time
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from itertools import groupby
# import matplotlib.pyplot as plt

'''
dRNA DNA barcode extraction.
looks for drop in signal and get's it as a segment

output structure:
fast5, readID, start, stop


'''
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)



def main():
    '''
    main function
    '''

    parser = MyParser(
        description="dRNA_segmenter - cut out adapter region of dRNA signal")
    #group = parser.add_mutually_exclusive_group()
    parser.add_argument("-s", "--signal",
                        help="Signal file")
    parser.add_argument("-c", "--start_col", type=int, default=1,
                        help="start column for signal")

    args = parser.parse_args()

    # print help if no arguments given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # arguments...put this into something better for Tansel
    file = args.signal         # signal file
    # sys.stderr.write("file:{}\n".format(file))
    w = 2000

    with open(file, 'rt') as s:
        for read in s:
            read = read.strip('\n')
            read = read.split('\t')
            readID = read[0]
            # roll_start_time = time.time()
            br_seg = b_roll(read, w, args.start_col)
            # roll_end_time = time.time() - roll_start_time
            # sys.stderr.write("B_roll time taken: {}\n".format(roll_end_time))
            # conv_start_time = time.time()
            bc_seg = b_conv(read, args.start_col)
            # conv_end_time = time.time() - conv_start_time
            # sys.stderr.write("B_conv time taken: {}\n".format(conv_end_time))
            # print(br_seg, bc_seg)
            print(readID, br_seg[0], br_seg[1], bc_seg[0], bc_seg[1], sep='\t')
            # uncomment to interactive plot
            # sig = scale_outliers(np.array([int(i) for i in read[args.start_col:]], dtype=int))
            # fig = plt.figure(1)
            # ax = fig.add_subplot(111)
            #
            # # Show segment lines
            # ax.axvline(x=br_seg[0], color='m')
            # ax.axvline(x=br_seg[1], color='m')
            # ax.axvline(x=bc_seg[0], color='g')
            # ax.axvline(x=bc_seg[1], color='g')
            #
            # plt.plot(sig, color='k')
            # plt.show()
            # plt.clf()


def b_roll(read, w, c):
    '''
    do b_roll
    '''
    sig = scale_outliers(np.array([int(i) for i in read[c:]], dtype=int))

    s = pd.Series(sig)
    t = s.rolling(window=w).mean()
    # This should be done better, or changed to median and benchmarked
    # Currently trained on mean segmented data
    mn = t.mean()
    std = t.std()
    # Trained on 0.5
    bot = mn - (std*0.5)

    # main algo

    begin = False
    # max distance for merging 2 segs
    seg_dist = 1500
    # max length of a seg
    hi_thresh = 200000
    # min length of a seg
    lo_thresh = 2000

    start = 0
    end = 0
    segs = []
    count = -1
    for i in t:
        count += 1
        if i < bot and not begin:
            start = count
            begin = True
        elif i < bot:
            end = count
        elif i > bot and begin:
            if segs and start - segs[-1][1] < seg_dist:
                segs[-1][1] = end
            else:
                segs.append([start, end])
            start = 0
            end = 0
            begin = False
        else:
            continue

    # offset = -1050
    # buff = 150
    offset = -1000
    buff = 0

    x, y = 0, 0

    for a, b in segs:
        if b - a > hi_thresh:
            continue
        if b - a < lo_thresh:
            continue
        x, y = a+offset, b+offset
        break
    # to be modified in next major re-training
    return x, y

def scale_outliers(squig):
    ''' Scale outliers to within m stdevs of median '''
    k = (squig > 0) & (squig < 1200)
    return squig[k]


def b_conv(read, c):
    '''
    do b_conv
    '''
    sig_limit = 20000
    readID = read[0]
    signal = read[1:sig_limit]
    der_size = 2001
    seg_size = 1001
    return sig_conv(readID, signal, der_size, seg_size)


def mad_scaling(signal):
    shift = np.median(signal)
    scale = np.median(np.abs(signal - shift))
    sd = np.std(signal)
    return ((signal - shift)/scale)

def der(s,win,order):
    '''
    inputs are:
    s: list or numpy array
    win: window size
    '''
    return savgol_filter(s, window_length=win, polyorder=2, deriv=order)

def sig_conv(read, signal, der_size, seg_size):
    # print ('processing',read,file=sys.stderr)
    try:
        s = np.array(signal).astype(int)
        s = mad_scaling(s)
        v = np.hstack([np.ones(s.shape), np.ones(s.shape) * - 1])
        conv = np.convolve (s, v, mode = 'valid')
        y = list(conv/1000)
        #valleys, peaks = turning_points(list(conv2))
        der1 = der(conv, der_size, 1)
        sign = np.asarray(der1 > 0).astype (int)
        segments = np.array([list(grp) for k, grp in groupby(sign)])
        #print (sign[:3])
        new_sign = []
        #new_sign.append (segments[0])
        for i in range(len(segments))[0:]:
            if len(segments[i]) <= seg_size and i >0:
                new_sign[-1] = new_sign[-1] + [new_sign[-1][0] for _ in segments[i]]
            else:
                new_sign.append(segments[i])
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
        elif new_segments[0][0] == 1 and len(new_segments[0])<1000: #no leader
            print(read,'no leader', sep='\t', file=sys.stderr)
            coors.append(pos)
            for i in new_segments:
                if (len(coors)) > 3:
                    break
                pos += len(i)
                coors.append(pos)
        else:
            for i in new_segments:
                if (len(coors)) > 3:
                    break
                pos += len(i)
                coors.append(pos)

        # print(coors)
        if len(coors) > 1:
            return coors[0], coors[1]
        else:
            return 0, 0
    except:
        raise
        print('error occured when processing' ,read, file=sys.stderr)
        return 0, 0





if __name__ == '__main__':
    main()
