#!/usr/bin/env python3
# coding: utf-8

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import sys
import os
from copy import deepcopy
import re
import csv
import time
import configparser
import h5py
import traceback
import math
import numpy as np
# from PIL import Image
import pyts
from pyts.image import MarkovTransitionField, GramianAngularField, RecurrencePlot
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from keras.models import load_model


'''

    James M. Ferguson (j.ferguson@garvan.org.au)
    Genomic Technologies
    Garvan Institute
    Copyright 2019

    Tansel Ersevas (t.ersevas@garvan.org.au)

    script description

    Deeplexicon: Demultiplex barcoded ONT direct-RNA sequencing reads

    ----------------------------------------------------------------------------
    version 0.0.0 - initial
    version 0.8.0 - CPU version Done
    version 0.9.0 - Fixed segment offset
    version 0.9.1 - added segment and squiggle output
    version 0.9.2 - separate segment output and code clean up
    version 1.0.0 - initial release
    version 1.1.0 - added submodules, splitting, and trining

    So a cutoff of: 0.4958776 for high accuradef read_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return(config)cy
    and another of 0.2943664 for high recovery

    TODO:
        - Remove leftover libraries
        - remove debug plots
        - Remove redundant code
        - create log files with information *****
        - add citation
        - create config file, for maintaining parity between training/dmuxing
        - load in ^ config for dmuxing


    ----------------------------------------------------------------------------
    MIT License

    Copyright (c) 2019 James M. Ferguson

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def print_verbose(message):
    '''verbose printing'''
    sys.stderr.write('info: %s\n' % message)

def print_err(message):
    '''error printing'''
    sys.stderr.write('error: %s\n' % message)

def _get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

def _check_available_devices():
    available_devices = _get_available_devices()
    print_verbose(available_devices)
    # Make sure requested GPUs are available or at least warn if they aren't
    return(TRUE)

def read_model(model_name):
    # model = load_model('saved_models/' + model_name)
    model = load_model(model_name) # as a path
    model.compile(loss='categorical_crossentropy',
                     optimizer=Adam(),
                     metrics=['accuracy'])
    return(model)


# TODO: this is messy, don't use lame globals like this
squiggle_max = 1199
squiggle_min = 1
input_cut = 72000 #potenitall need to be configurable
image_size = 224
num_classes = 4
window = 2000

def main():
    '''
    Main function
    '''
    VERSION = "1.1.0"

    parser = MyParser(
        description="DeePlexiCon - Demultiplex barcoded ONT direct-RNA sequencing reads",
        epilog="Citation: enter publication here...",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subcommand = parser.add_subparsers(help='subcommand --help for help messages', dest="command")

    # main options for base level checks and version output
    parser.add_argument("--version", action='version', version="Deeplexicon version: {}".format(VERSION),
                        help="Prints version")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbose output [v/vv/vvv]")

    # sub-module for dmux command
    dmux = subcommand.add_parser('dmux', help='demultiplex dRNA reads',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # dmux sub-module options
    dmux.add_argument("-p", "--path", nargs='+',
                        help="Top path(s) of fast5 files to dmux")
    dmux.add_argument("-f", "--form", default="multi", choices=["multi", "single"],
                        help="Multi or single fast5s")
    dmux.add_argument("-s", "--threshold", type=float, default=0.50,
                        help="probability threshold - 0.5 hi accuracy / 0.3 hi recovery")
    dmux.add_argument("-m", "--model",
                        help="Trained model name to use")
    dmux.add_argument("--squiggle", default=False,
                        help="dump squiggle data into this .tsv file")
    dmux.add_argument("--segment", default=False,
                        help="dump segment data into this .tsv file")
    dmux.add_argument("-b", "--batch_size", type=int, default=4000,
                        help="batch size - for single fast5s")
    dmux.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbose output [v/vv/vvv]")

    # sub-module for split command
    split = subcommand.add_parser('split', help='split a fastq file into barcode categories',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # split sub-module options
    split.add_argument("-i", "--input",
                        help="deeplexicon dmux output tsv file")
    split.add_argument("-q", "--fastq",
                        help="single combined fastq file")
    split.add_argument("-o", "--output",
                        help="output path")
    split.add_argument("-s", "--sample", default="dmux_",
                        help="sample name to append to file names")
    split.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbose output [v/vv/vvv]")

    # sub-module for train command
    train = subcommand.add_parser('train', help='train a demultiplexing model',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # train sub-module options
    # take all fast5s, for each read, check train_truth and convert
    # repeat for test_truth - could add in validation set too using dmux?
    # data, settings, model output, tmp?,
    train.add_argument('-p', '--path', nargs='+',
                        help="Input path(s) of all used fast5s")
    train.add_argument('-t', '--train_truth', nargs='+',
                        help="Traiing truth set(s) in one-hot format eg: readID, 0, 0, 1, 0 for barcode 3 of 4 ")
    train.add_argument('-s', '--test_truth', nargs='+',
                        help="Testing truth set(s) in one-hot format eg: readID, 0, 0, 1, 0 for barcode 3 of 4 ")
    train.add_argument('-u', '--val_truth', nargs='+',
                        help="Validation truth set(s) in one-hot format eg: readID, 0, 0, 1, 0 for barcode 3 of 4 ")
    train.add_argument('-n', '--network', default="ResNet20",
                        help="Network to use (see table in docs)")
    train.add_argument('--net_version', type=int, default=2,
                        help="Network version to use (see table in docs)")
    train.add_argument('-e', '--epochs', type=int, default=40,
                        help="epochs to run")
    train.add_argument('-x', '--prefix', default="model",
                        help="prefix used to name model")
    train.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbose output [v/vv/vvv]")

    # sub-module for squig command
    squig = subcommand.add_parser('squig', help='extract/segment squiggles - no dmux',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # squig sub-module options
    squig.add_argument("--squiggle",
                        help="dump squiggle data into this .tsv file")
    squig.add_argument("--segment",
                        help="dump segment data into this .tsv file")
    squig.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbose output [v/vv/vvv]")


    # collect args
    args = parser.parse_args()

    # print help if no arguments given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.verbose > 0:
        print_verbose("Verbose mode active - dumping info to stderr")
        print_verbose("DeePlexiCon: {}".format(VERSION))
        print_verbose("arg list: {}".format(args))
        if tf.test.gpu_device_name():
            print_verbose('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print_verbose("Please install GPU version of TF")


    # Ensure non-command use is exited before this point
    # Perfect time to do arg checks before pipeline calls
    if args.command == "dmux":
        dmux_pipeline(args)
    elif args.command == "split":
        split_pipeline(args)
    elif args.command == "train":

        def get_available_gpus():
            local_device_protos = device_lib.list_local_devices()
            return [x.name for x in local_device_protos if x.device_type == 'GPU']

        gpu_list = get_available_gpus()
        if len(gpu_list) < 1:
            print("No GPU detected. Please ensure CUDA and cuDNN are set up")
            sys.exit(1)
        print("Num GPUs Available: ", len(gpu_list))
        print("Only single GPU mode available, using device: {}".format(gpu_list[0]))

        train_pipeline(args)
    elif args.command == "squig":
        squig_pipeline(args)
    else:
        print_err("command: {} not recognised".format(args.command))
        parser.print_help(sys.stderr)
        sys.exit(1)

    # done!

    # # TODO: sub-module
    # if args.squiggle:
    #     squig_file = args.squiggle
    #     with open(squig_file, 'a') as f:
    #         f.write("{}\t{}\n".format("ReadID", "signal_pA"))
    # else:
    #     squig_file = ''
    # # TODO: sub-module
    # if args.segment:
    #     seg_file = args.segment
    #     with open(seg_file, 'a') as f:
    #         f.write("{}\t{}\t{}\n".format("ReadID", "start", "stop"))
    # else:
    #     seg_file = ""

    # Globals
    # TODO: sub-module
    # if args.config:
    #     config = read_config(args.config) #TODO check config read error

    # gpu settings
    # Devices
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # os.environ["CUDA_VISIBLE_DEVICES"] = config[deeplexicon][gpu_list] if args.config else args.gpu_list

    # do check devices are available, else throw and error


    # DMUX sub-module
    # main logic
def dmux_pipeline(args):
    '''
    pipeline for dmuxing fast5 files
    '''
    # read model
    if not args.model:
        print_err("dmux requires a trained model file path")
        sys.exit(1)
    model = read_model(args.model)
    # make this an optional config input for custom barcode sets
    barcode_out = {0: "bc_1",
                   1: "bc_2",
                   2: "bc_3",
                   3: "bc_4",
                   None: "unknown"
                   }
    labels = []
    images = []
    fast5s = {}
    stats = ""
    seg_dic = {}
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format("fast5", "ReadID", "Barcode", "Confidence Interval", "P_bc_1", "P_bc_2", "P_bc_3", "P_bc_4"))
    # for file in input...
    # TODO: sub-module
    for dirpath, dirnames, files in os.walk(dmux.path):
        for fast5 in files:
            if fast5.endswith('.fast5'):
                fast5_file = os.path.join(dirpath, fast5)
                if args.form == "single":
                    #everthing below this, send off in batches of N=args.batch_size
                    # The signal extraction and segmentation can happen in the first step
                    # read fast5 files
                    readID, seg_signal = get_single_fast5_signal(fast5_file, window, squig_file, seg_file)
                    if not seg_signal:
                        print_err("Segment not found for:\t{}\t{}".format(fast5_file, readID))
                        continue
                    # convert
                    sig = np.array(seg_signal, dtype=float)
                    img = convert_to_image(sig)
                    labels.append(readID)
                    fast5s[readID] = fast5
                    images.append(img)
                    # classify
                    if len(labels) >= args.batch_size:
                        C = classify(model, labels, np.array(images), False, args.threshold)
                        # save to output
                        for readID, out, c, P in C:
                            prob = [round(float(i), 6) for i in P]
                            cm = round(float(c), 4)
                            if args.verbose:
                                print_verbose("cm is: {}".format(cm))
                            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(fast5s[readID], readID, barcode_out[out], cm, prob[0], prob[1], prob[2], prob[3]))
                        labels = []
                        images = []
                        fast5s = {}

                # TODO: sub-module
                elif args.form == "multi":
                    #everthing below this, send off in batches of N=args.batch_size
                    # The signal extraction and segmentation can happen in the first step
                    # read fast5 files
                    seg_signal = get_multi_fast5_signal(fast5_file, window, squig_file, seg_file)
                    sig_count = 0
                    for readID in seg_signal:
                        # convert
                        img = convert_to_image(np.array(seg_signal[readID], dtype=float))
                        labels.append(readID)
                        images.append(img)
                        fast5s[readID] = fast5
                        sig_count += 1
                        # TODO: sub-module
                        if len(labels) >= args.batch_size:
                            C = classify(model, labels, np.array(images), False, args.threshold)
                            # save to output
                            for readID, out, c, P in C:
                                prob = [round(float(i), 6) for i in P]
                                cm = round(float(c), 4)
                                if args.verbose:
                                    print_verbose("cm is: {}".format(cm))
                                print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(fast5s[readID], readID, barcde_out[out], cm, prob[0], prob[1], prob[2], prob[3]))
                            labels = []
                            images = []
                            fast5s = {}
                        elif args.verbose:
                            print_verbose("analysing sig_count: {}/{}".format(sig_count, len(seg_signal)))
                        else:
                            blah = 0 # clean
    #finish up
    # TODO: sub-module
    C = classify(model, labels, np.array(images), False, args.threshold)
    # save to output
    for readID, out, c, P in C:
        prob = [round(float(i), 6) for i in P]
        cm = round(float(c), 4)
        if args.verbose:
            print_verbose("cm is: {}".format(cm))
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(fast5s[readID], readID, barcode_out[out], cm, prob[0], prob[1], prob[2], prob[3]))
    labels = []
    images = []
    fast5s = {}


    # final report/stats
    # print stats
    return


def split_pipeline(args):
    '''
    split fastq file using dmux file
    '''
    def _get_reads(filename):
        '''
        Build dmux dic
        '''
        dic = {}
        bc_set = set()
        head = True
        with open(filename, 'rt') as f:
            for l in f:
                if head:
                    head = False
                    continue
                l = l.strip('\n')
                l = l.split('\t')
                dic[l[1]] = l[2]
                if l[2] not in bc_set:
                    bc_set.add(l[2])
        return dic, bc_set


    def _split_fastq(read_bcs, bc_set, fastq, output, sample):
        '''
        split fastq into multiple fastq
        '''
        dic = {}
        c = 0
        P = False
        for i in bc_set:
            file = os.path.join(output, "{}_{}.fastq".format(sample, i))
            dic[i] = open(file, 'w')
        with open(fastq, 'rt') as f:
            for l in f:
                c += 1
                ln = l.strip('\n')
                if c == 1:
                    ln = ln.split(' ')
                    readID = ln[0][1:]
                    if readID in read_bcs:
                        bc = read_bcs[readID]
                        P = True
                        dic[bc].write(l)
                elif c < 5 and P:
                    dic[bc].write(l)
                if c >= 4:
                    P = False
                    c = 0

        for i in list(dic.keys()):
            dic[i].close

        return

    # run split pipeline
    read_bcs, bc_set = _get_reads(args.input)

    _split_fastq(read_bcs, bc_set, args.fastq, args.output, args.sample)

    return


# file handling and segmentation

def get_single_fast5_signal(read_filename, w, squig_file, seg_file):
    '''
    open sigle fast5 file and extract information
    '''

    # get readID and signal
    f5_dic = read_single_fast5(read_filename)
    if not f5_dic:
        print_err("Signal not extracted from: {}".format(read_filename))
        return 0, 0
    # segment on raw
    readID = f5_dic['readID']
    signal = f5_dic['signal']
    seg = dRNA_segmenter(readID, signal, w)
    if not seg:
        print_verbose("No segment found - skipping: {}".format(readID))
        return 0, 0
    # convert to pA
    pA_signal = convert_to_pA(f5_dic)
    if squig_file:
        with open(squig_file, 'a') as f:
            f.write("{}\t{}\n".format(readID, "\t".join(pA_signal)))
    if seg_file:
        with open(seg_file, 'a') as f:
            f.write("{}\t{}\t{}\n".format(readID, seg[0], seg[1]))
    # return signal/signals
    return readID, pA_signal[seg[0]:seg[1]]


def get_multi_fast5_signal(read_filename, w, squig_file, seg_file, train=False):
    '''
    open multi fast5 files and extract information
    '''
    pA_signals = {}
    seg_dic = {}
    f5_dic = read_multi_fast5(read_filename, reads=train)
    seg = 0
    sig_count = 0
    for read in f5_dic:
        sig_count += 1
        print_verbose("reading sig_count: {}/{}".format(sig_count, len(f5_dic)))
        # get readID and signal
        readID = f5_dic[read]['readID']
        signal = f5_dic[read]['signal']

        # segment on raw
        seg = dRNA_segmenter(readID, signal, w)
        if not seg:
            seg = 0
            continue
        # convert to pA
        pA_signal = convert_to_pA(f5_dic[read])
        if squig_file:
            with open(squig_file, 'a') as f:
                f.write("{}\t{}\n".format(readID, "\t".join(pA_signal)))
        if seg_file:
            with open(seg_file, 'a') as f:
                f.write("{}\t{}\t{}\n".format(readID, seg[0], seg[1]))
        pA_signals[readID] = pA_signal[seg[0]:seg[1]]
        seg_dic[readID] = seg
    # return signal/signals
    return pA_signals


def read_single_fast5(filename):
    '''
    read single fast5 file and return data
    '''
    f5_dic = {'signal': [], 'readID': '', 'digitisation': 0.0,
              'offset': 0.0, 'range': 0.0, 'sampling_rate': 0.0}

    # open fast5 file
    try:
        hdf = h5py.File(filename, 'r')
    except:
        traceback.print_exc()
        print_err("extract_fast5():fast5 file failed to open: {}".format(filename))
        f5_dic = {}
        return f5_dic
    try:
        c = list(hdf['Raw/Reads'].keys())
        # for col in hdf['Raw/Reads/'][c[0]]['Signal'][()]:
        #     f5_dic['signal'].append(int(col))
        f5_dic['signal'] = hdf['Raw/Reads/'][c[0]]['Signal'][()]

        f5_dic['readID'] = hdf['Raw/Reads/'][c[0]].attrs['read_id'].decode()
        f5_dic['digitisation'] = hdf['UniqueGlobalKey/channel_id'].attrs['digitisation']
        f5_dic['offset'] = hdf['UniqueGlobalKey/channel_id'].attrs['offset']
        f5_dic['range'] = float("{0:.2f}".format(hdf['UniqueGlobalKey/channel_id'].attrs['range']))
        f5_dic['sampling_rate'] = hdf['UniqueGlobalKey/channel_id'].attrs['sampling_rate']

    except:
        traceback.print_exc()
        print_err("extract_fast5():failed to extract events or fastq from: {}".format(filename))
        f5_dic = {}

    return f5_dic


def read_multi_fast5(filename, reads=False):
    '''
    read multifast5 file and return data
    '''
    f5_dic = {}
    with h5py.File(filename, 'r') as hdf:
        for read in list(hdf.keys()):
            try:
                if reads:
                    if hdf[read]['Raw'].attrs['read_id'].decode() not in reads:
                        continue
                f5_dic[read] = {'signal': [], 'readID': '', 'digitisation': 0.0,
                'offset': 0.0, 'range': 0.0, 'sampling_rate': 0.0}

                f5_dic[read]['signal'] = hdf[read]['Raw/Signal'][()]
                f5_dic[read]['readID'] = hdf[read]['Raw'].attrs['read_id'].decode()
                f5_dic[read]['digitisation'] = hdf[read]['channel_id'].attrs['digitisation']
                f5_dic[read]['offset'] = hdf[read]['channel_id'].attrs['offset']
                f5_dic[read]['range'] = float("{0:.2f}".format(hdf[read]['channel_id'].attrs['range']))
                f5_dic[read]['sampling_rate'] = hdf[read]['channel_id'].attrs['sampling_rate']
            except:
                traceback.print_exc()
                print_err("extract_fast5():failed to read readID: {}".format(read))
    return f5_dic


def dRNA_segmenter(readID, signal, w):
    '''
    segment signal/s and return coords of cuts
    '''
    def _scale_outliers(squig):
        ''' Scale outliers to within m stdevs of median '''
        k = (squig > 0) & (squig < 1200)
        return squig[k]


    sig = _scale_outliers(np.array(signal, dtype=int))

    s = pd.Series(sig)
    t = s.rolling(window=w).mean()
    # This should be done better, or changed to median and benchmarked
    # Currently trained on mean segmented data
    # Make it an argument for user to choose in training/dmux and config
    mn = t.mean()
    std = t.std()
    # Trained on 0.5
    bot = mn - (std*0.5)

    # main algo
    # TODO: add config for these for users to fiddle with
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
    # half the window - probs should be offset = w / 2
    offset = -1000
    buff = 0

    x, y = 0, 0

    for a, b in segs:
        if b - a > hi_thresh:
            continue
        if b - a < lo_thresh:
            continue
        x, y = a, b

        # to be modified in next major re-training
        return [x+offset-buff, y+offset+buff]
        break
    print_verbose("dRNA_segmenter: no seg found: {}".format(readID))
    return 0


def convert_to_pA(d):
    '''
    convert raw signal data to pA using digitisation, offset, and range
    float raw_unit = range / digitisation;
    for (int32_t j = 0; j < nsample; j++) {
        rawptr[j] = (rawptr[j] + offset) * raw_unit;
    }
    '''
    digitisation = d['digitisation']
    range = d['range']
    offset = d['offset']
    raw_unit = range / digitisation
    new_raw = []
    for i in d['signal']:
        j = (i + offset) * raw_unit
        new_raw.append("{0:.2f}".format(round(j,2)))
    return new_raw


def pyts_transform(transform, data, image_size, show=False, cmap='rainbow', img_index=0):
    try:
        t_start=time.time()
        X_transform = transform.fit_transform(data)
        if (show):
            plt.figure(figsize=(4, 4))
            plt.grid(b=None)
            plt.imshow(X_transform[0], cmap=cmap, origin='lmtfower')
            plt.savefig(transform.__class__.__name__ + "_image_" + str(img_index) + ".svg", format="svg")
            plt.show()
        return(X_transform)
    except Exception as e:
        print_err(str(e))
        return([])


def mtf_transform(data, image_size=500, show=False, img_index=0):
    transform = MarkovTransitionField(image_size)
    return(pyts_transform(transform, data, image_size=image_size, show=show, cmap='rainbow', img_index=img_index))

def rp_transform(data, image_size=500 ,show=False ,img_index=0):
    # RP transformationmtf
    transform = RecurrencePlot(dimension=1,
                    threshold='percentage_points',
                    percentage=30)
    return(pyts_transform(transform, data, image_size=image_size, show=show, cmap='binary', img_index=img_index))

def gasf_transform(data, image_size=500, show=False, img_index=0):
    # GAF transformation
    transform = GramianAngularField(image_size, method='summation')
    return(pyts_transform(transform, data, image_size=image_size, show=show, cmap='rainbow', img_index=img_index))

def gadf_transform(data, image_size=500, show=False ,img_index=0):
    # GAF transformation
    transform = GramianAngularField(image_size, method='difference')
    return(pyts_transform(transform, data, image_size=image_size, show=show, cmap='rainbow', img_index=img_index))


def labels_for(a_file_name):
    segments=re.split(r'[_\-\.]+', a_file_name)
    return(segments)

def max_in_sequence(sequence):
    return(max(np.amax([list(d.values()) for d in sequence]), 0.01))

def compress_squiggle(squiggle, compress_factor):
    squiggle_len = len(squiggle)
    rem = squiggle_len % compress_factor
    if rem > 0:
        return(np.mean(squiggle[0:squiggle_len - rem].reshape(-1,compress_factor), axis=1))
    return(squiggle)

def convert_to_image(signal):
   transformed_squiggle = gasf_transform(signal.reshape(1,-1), image_size=image_size, show=False)
   return(transformed_squiggle)


def confidence_margin(npa):
    sorted = np.sort(npa)[::-1]    #return sort in reverse, i.e. descending
    # sorted = np.sort(npa)   #return sort in reverse, i.e. descending
    d = sorted[0] - sorted[1]
    return(d)

def classify(model, labels, image, subtract_pixel_mean, threshold):
    input_shape = image.shape[1:]
    # x = image.astype('float32') / 255
    x = image.astype('float32') + 1
    x = x / 2

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_mean = np.mean(x, axis=0)
        x -= x_mean
    x=[x]
    y = model.predict(x, verbose=0)
    res = []
    for i in range(len(y)):
        cm = confidence_margin(y[i])
        if y[i][np.argmax(y[i])] >= threshold:
            res.append([labels[i], np.argmax(y[i]), cm, y[i]])
        else:
            res.append([labels[i], None, cm, y[i]])
    return res


def train_pipeline(args):
    '''
    train a new dmux model

    Defines a ResNet on the nanopore dataset.

    ResNet v1
    [a] Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf

    ResNet v2
    [b] Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf

    Usage

    from resnet import train_model #and optional resnet_package_versions

    # train_model(run_name, net_type,version,  epochs,  x_train, y_train, x_test, y_test,
    #          gpus=1,per_gpu_batch_size=16,tensorboard_output=None, data_augmentation = False, subtract_pixel_mean = False, verbose=0)


    history=train_model(run, "ResNet20",2,  epochs,  x_train, y_train, x_test, y_test,
               gpus=gpus,per_gpu_batch_size=16

    '''

    def resnet_package_versions():
        print("Tensorflow   version :",tf.__version__)
        print("Keras        version :",keras.__version__)

    def lr_schedule(epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 10, 20, 30, 50 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 50:
            lr *= 0.5e-3
        elif epoch > 45:
            lr *= 1e-3
        elif epoch > 30:
            lr *= 1e-2
        elif epoch > 15:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr


    # Training parameters
    def depth_for(nn_name, version):
         # Model version
        # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)

        # Model parameter
        # ----------------------------------------------------------------------------
        #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
        # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
        #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
        # ----------------------------------------------------------------------------
        # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
        # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
        # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
        # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
        # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
        # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
        # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
        # ---------------------------------------------------------------------------
        nn_table={'ResNet20':[3,2],'ResNet32':[5,None],'ResNet44':[7,None],'ResNet56':[9,6],
                  'ResNet110':[18,12],'ResNet164':[27,18],'ResNet1001':[None,111]}

        n = nn_table[nn_name][version-1]

        # Computed depth from supplied model parameter n
        if version == 1:
            depth = n * 6 + 2
        elif version == 2:
            depth = n * 9 + 2
        return(depth)

    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      data_format='channels_first',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            print("Convolution name= ",x.name, " numfilters=", num_filters, " kernel_size=", kernel_size, " strides=", strides)
            if batch_normalization:
                x = BatchNormalization()(x)
                print("Batch normalisation")
            if activation is not None:
                x = Activation(activation)(x)
                print("Activation")
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
                print("Batch normalisation")
            if activation is not None:
                x = Activation(activation)(x)
                print("Activation")
            x = conv(x)
            print("Convolution name= ",x.name, " numfilters=", num_filters, " kernel_size=", kernel_size, " strides=", strides)
        conv.name=conv.name+'_'+str(kernel_size)+'x'+str(kernel_size)+'_'+str(num_filters)+'_'+str(strides)
        return x


    def resnet_v1(input_shape, depth, num_classes=4):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model


    def resnet_v2(input_shape, depth, num_classes=4):
        """ResNet Version 2 Model builder [b]

        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = resnet_layer(inputs=inputs,
                         num_filters=num_filters_in,
                         conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # bottleneck residual unit
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_in,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def train_model(run_name, net_type,version,  epochs,  x_train, y_train, x_test, y_test,
               gpus=1,per_gpu_batch_size=16,tensorboard_output=None, data_augmentation = False,
               subtract_pixel_mean = False, verbose=0, x_val=False, y_val=False):

        batch_size=per_gpu_batch_size*gpus  # multiply by number of GPUs


    # Subtracting pixel mean improves accuracy


        depth=depth_for(net_type,version)
        model_type = 'UResNet%dv%d' % (depth, version)
        input_shape = x_train.shape[1:]
        if depth==None:
            return(None) # Means Invalid Network Type or Version

        if version == 2:
            model = resnet_v2(input_shape=input_shape, depth=depth)
        else:
            model = resnet_v1(input_shape=input_shape, depth=depth)
        model.summary()
        #    plot_model(model, to_file='model_plot.svg', show_shapes=True, show_layer_names=True)
        # if gpus>1:
        #     model = multi_gpu_model(model, gpus=gpus, cpu_merge=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])

        # Prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = run_name+'_dRNA_nanopore_%s_model.{epoch:03d}.h5' % model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=verbose,
                                     save_best_only=True)

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        if tensorboard_output==None:
            callbacks = [checkpoint, lr_reducer, lr_scheduler]
        else:
            tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_output+"/{}".format(time.time()),
                          histogram_freq=20, write_graph=True)
            callbacks = [tensorboard, checkpoint, lr_reducer, lr_scheduler]
        # Run training, with or without data augmentation.
        if not data_augmentation:
            history=model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
        #              steps_per_epoch=batch_size,
                      validation_data=(x_test, y_test),
        #              validation_steps=1,
                      shuffle=True,
                      callbacks=callbacks)
        else:
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataseargs.epochs=t
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=0,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=False,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            history=model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                validation_data=(x_test, y_test),
                                epochs=epochs, verbose=verbose, workers=4,
                                steps_per_epoch=batch_size,
                                callbacks=callbacks)
        if x_val is not False:
            scores = model.evaluate(x_val, y_val, verbose=1)
            print('Validation loss: {} accuracy: {}'.format(scores[0], scores[1]))

        return(history)



    # actual pipeline
    # get test/train IDs
    train_truth_dic = {}
    train_readIDs = set()
    test_readIDs = set()
    val_readIDs = set()
    for train_file in args.train_truth:
        with open(train_file, 'rt') as tt:
            for l in tt:
                l = l.strip("\n")
                l = l.split("\t")
                if len(l) < 3:
                    l = l[0].split(",")
                readID = l[0]
                train_truth_dic[readID] = np.array([float(i) for i in l[1:]], dtype=float)
    train_readIDs = set(train_truth_dic.keys())

    test_truth_dic = {}
    for test_file in args.test_truth:
        with open(test_file, 'rt') as tt:
            for l in tt:
                l = l.strip("\n")
                l = l.split("\t")
                if len(l) < 3:
                    l = l[0].split(",")
                readID = l[0]
                test_truth_dic[readID] = np.array([float(i) for i in l[1:]], dtype=float)
    test_readIDs = set(test_truth_dic.keys())

    if args.val_truth:
        val_truth_dic = {}
        for val_file in args.val_truth:
            with open(val_file, 'rt') as tt:
                for l in tt:
                    l = l.strip("\n")
                    l = l.split("\t")
                    if len(l) < 3:
                        l = l[0].split(",")
                    readID = l[0]
                    val_truth_dic[readID] = np.array([float(i) for i in l[1:]], dtype=float)
        val_readIDs = set(val_truth_dic.keys())

    all_reads = set()
    all_reads.update(train_readIDs)
    all_reads.update(test_readIDs)
    all_reads.update(val_readIDs)
    # read fast5s and convert them to images
    labels = []
    images = []
    fast5s = {}
    seg_dic = {}
    train_labels = []
    train_images = []
    train_fast5s = {}
    test_labels = []
    test_images = []
    test_fast5s = {}
    val_labels = []
    val_images = []
    val_fast5s = {}
    train_count = 0
    test_count = 0
    val_count = 0
    for p in args.path:
        for dirpath, dirnames, files in os.walk(p):
            for fast5 in files:
                if fast5.endswith('.fast5'):
                    fast5_file = os.path.join(dirpath, fast5)
                    print(fast5)
                    seg_signal = get_multi_fast5_signal(fast5_file, window, False, False, train=all_reads)
                    for readID in seg_signal:
                        if readID in train_readIDs:
                            img = convert_to_image(np.array(seg_signal[readID], dtype=float))
                            train_labels.append(train_truth_dic[readID])
                            train_images.append(img)
                            train_fast5s[readID] = fast5
                            train_count += 1
                        elif readID in test_readIDs:
                            img = convert_to_image(np.array(seg_signal[readID], dtype=float))
                            test_labels.append(test_truth_dic[readID])
                            test_images.append(img)
                            test_fast5s[readID] = fast5
                            test_count += 1
                        elif readID in val_readIDs:
                            img = convert_to_image(np.array(seg_signal[readID], dtype=float))
                            val_labels.append(val_truth_dic[readID])
                            val_images.append(img)
                            val_fast5s[readID] = fast5
                            val_count += 1

                        else:
                            continue

    gpus = 1
    batch_control = 8
    if args.val_truth:
        ret=train_model(args.prefix, args.network, args.net_version, args.epochs,
                        np.array(train_images), np.array(train_labels),
                        np.array(test_images), np.array(test_labels),
                        gpus=gpus,per_gpu_batch_size=batch_control,
                        x_val=np.array(val_images), y_val=np.array(val_labels))
    else:
        ret=train_model(args.prefix, args.network, args.net_version, args.epochs,
                        np.array(train_images), np.array(train_labels),
                        np.array(test_images), np.array(test_labels),
                        gpus=gpus,per_gpu_batch_size=batch_control)


    return


def squig_pipeline(args):
    '''
    extract/segment squiggles
    '''

    return


if __name__ == '__main__':
    main()
