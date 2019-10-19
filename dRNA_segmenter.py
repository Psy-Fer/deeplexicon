import numpy as np
import matplotlib.pyplot as plt
import sys, gzip
import pandas as pd
import argparse


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
    parser.add_argument("-m", "--seq_sum",
                        help="sequencing_summary file")
    parser.add_argument("-w", "--window", type=int, default=2000,
                        help="Window size")
    parser.add_argument("-i", "--input_type", default="pA", choices=["pA", "raw"],
                        help="Signal file input type")
    parser.add_argument("-c", "--start_col", type=int, default="4",
                        help="start column for signal")

    args = parser.parse_args()

    # print help if no arguments given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # arguments...put this into something better for Tansel
    sig = args.signal         # signal file
    SS = args.seq_sum          # Seq_sum.txt
    w = args.window      # window size
    # val_reads = sys.argv[3]
    # error = int(sys.argv[3])
    # thresh = int(sys.argv[4])
    read_dic = read_SS(SS)

    if sig.endswith('.gz'):
        f_read = dicSwitch('gz')
    else:
        f_read = dicSwitch('norm')
    with f_read(sig, 'rt') as s:
        for read in s:
            read = read.strip('\n')
            read = read.split('\t')
            f5 = read[0]
            if args.input_type == "pA":
                sig = scale_outliers(np.array([float(i) for i in read[args.start_col:]], dtype=float))

            elif args.input_type == "raw":
                sig = scale_outliers(np.array([int(i) for i in read[args.start_col:]], dtype=int))

            # HL data format
            # readID, signal = read.split()
            # sig = scale_outliers(np.array([int(i) for i in signal.split(',')], dtype=int))

            s = pd.Series(sig)
            t = s.rolling(window=w).mean()
            mn = t.mean()
            std = t.std()
            # might need to tighten the bounds a little more
            # top = mn + (std*0.5)
            bot = mn - (std*0.5)

            # main algo

            begin  = False
            seg_dist = 1500
            hi_thresh = 200000
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

            x, y = 0, 0
            for a, b in segs:
                if b - a > hi_thresh:
                    continue
                if b - a < lo_thresh:
                    continue
                x, y = a, b
                print "{}\t{}\t{}\t{}".format(f5, read_dic[f5], x, y)
                break





def scale_outliers(squig):
    ''' Scale outliers to within m stdevs of median '''
    k = (squig > 0) & (squig < 1200)
    return squig[k]


def dicSwitch(i):
    '''
    A switch to handle file opening and reduce duplicated code
    '''
    open_method = {
        "gz": gzip.open,
        "norm": open
    }
    return open_method[i]


def read_SS(seq_sum):
    '''
    Get read IDs
    '''
    head = True
    files = {}
    if seq_sum.endswith('.gz'):
        f_read = dicSwitch('gz')
    else:
        f_read = dicSwitch('norm')
    with f_read(seq_sum, 'rt') as sz:
        for line in sz:
            if head:
                head = False
                continue
            line = line.strip('\n')
            line = line.split()
            files[line[0]] = line[1]
    return files


if __name__ == '__main__':
    main()
