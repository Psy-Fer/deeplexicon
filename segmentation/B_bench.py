import numpy as np
import sys, os
import pandas as pd
import argparse
import time
from scipy.signal import medfilt


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
    parser.add_argument("-c", "--start_col", type=int, default="4",
                        help="start column for signal")

    args = parser.parse_args()

    # print help if no arguments given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # arguments...put this into something better for Tansel
    file = args.signal         # signal file
    w = 2000

    roll_start_time = time.time()
    b_roll(file, w, args.start_col)
    roll_end_time = time.time() - roll_start_time
    sys.stderr.write("B_roll time taken: {}\n".format(roll_end_time))
    conv_start_time = time.time()
    b_conv(file, args.start_col)
    conv_end_time = time.time() - conv_start_time
    sys.stderr.write("B_conv time taken: {}\n".format(conv_end_time))


def b_roll(file, w, c):
    with open(file, 'rt') as s:
        for read in s:
            read = read.strip('\n')
            read = read.split('\t')
            readID = read[1]

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
            print(readID ,x, y, sep="\t")

def b_conv(file, c):
    with open(file, 'rt') as s:
        for read in s:
            read = read.strip('\n')
            read = read.split('\t')
            readID = read[1]
            sig = [int(i) for i in read[c:]]
            raw_signal = np.array(list(map (int, sig[0:20000])))
            signal = mad_scaling(raw_signal)
            x = mad_scaling(list(range(len(signal))))
            v = np.hstack([np.ones(signal.shape), np.ones(signal.shape) * - 1])
            conv = np.convolve(medfilt (signal,2001), v, mode='valid')
            y = list (conv/1000)
            valleys, peaks = turning_points(list(conv))

            begin, end = '',''
            try:
                if peaks[0]>=1000:
                    begin = valleys[0]
                    end = peaks[0]

                else:
                    begin = valleys[0]
                    end = peaks[1]

                print(readID,begin,end,sep='\t')
            except:
                continue



def scale_outliers(squig):
    ''' Scale outliers to within m stdevs of median '''
    k = (squig > 0) & (squig < 1200)
    return squig[k]



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








if __name__ == '__main__':
    main()
