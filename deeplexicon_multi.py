#!/usr/bin/env python3
# coding: utf-8

from deeplexicon_sub import *
from multiprocessing import Pool

'''

    James M. Ferguson (j.ferguson@garvan.org.au)
    Genomic Technologies
    Garvan Institute
    Copyright 2019

    Tansel Ersevas (t.ersevas@garvan.org.au)

    Leszek Pryszcz (lpryszcz@crg.es)

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
    version 1.2.0 - segmentation ~10x faster; added multiprocessing via deeplexicon_multi.py (only for multi_fast5 files)

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

def main():
    '''
    Main function
    '''
    VERSION = "1.2.0"

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
    dmux.add_argument('--threads', default=1, type=int, help="Number of threads to use [%(default)s]")
    dmux.add_argument("-p", "--path", nargs='+',
                        help="Top path(s) of fast5 files to dmux")
    #dmux.add_argument("-f", "--form", default="multi", choices=["multi", "single"],
    #                    help="Multi or single fast5s")
    dmux.add_argument("-s", "--threshold", type=float, default=0.50,
                        help="probability threshold - 0.5 hi accuracy / 0.3 hi recovery")
    dmux.add_argument("-m", "--model",
                        help="Trained model name to use")
    dmux.add_argument('-N', '--Number', type=int, default=4,
                        help="Number of barcodes to dmux. controls header for custom models")
    dmux.add_argument("-g", "--gpu", action="store_true",
                        help="Use GPU if available - experimental")
    dmux.add_argument("--squiggle", default=False,
                        help="dump squiggle data into this .tsv file")
    dmux.add_argument("--segment", default=False,
                        help="dump segment data into this .tsv file")
    dmux.add_argument("-b", "--batch_size", type=int, default=1000,
                        help="batch size - for single fast5s")
    dmux.add_argument('-t', '--test', type=int,
                        help="test with -t number of reads")
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
    train.add_argument('-N', '--Number', type=int,
                        help="Number of barcodes to train. Should be auto detected, but set to check")
    train.add_argument('-n', '--network', default="ResNet20",
                        help="Network to use (see table in docs)")
    train.add_argument('--net_version', type=int, default=2,
                        help="Network version to use (see table in docs)")
    train.add_argument('-e', '--epochs', type=int, default=40,
                        help="epochs to run")
    train.add_argument('-b', '--batch_size', type=int, default=8,
                        help="Controls how much data is loaded into the GPU at a time. ~8 for 4GB cards, ~16 for >8GB")
    train.add_argument('-x', '--prefix', default="model",
                        help="prefix used to name model")
    train.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbose output [v/vv/vvv]")

    # sub-module for squig command
    squig = subcommand.add_parser('squig', help='extract/segment squiggles - no dmux',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # squig sub-module options
    squig.add_argument("-p", "--path", nargs='+',
                        help="Top path(s) of fast5 files to dmux")
    squig.add_argument("-f", "--form", default="multi", choices=["multi", "single"],
                        help="Multi or single fast5s (multi only for squig module)")
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
            print_verbose("GPU detected!!!")
            print_verbose("Default GPU Device: {}".format(tf.test.gpu_device_name()))
        else:
            print_verbose("Please install GPU version of TF:")
            print_verbose("> pip3 uninstall tensorflow")
            print_verbose("> pip3 install tensorflow-gpu=1.13.1")


    # Ensure non-command use is exited before this point
    # Perfect time to do arg checks before pipeline calls
    if args.command == "dmux":
        if args.gpu:
            if tf.test.gpu_device_name():
                print_verbose("GPU detected!!!")
                print_verbose("Default GPU Device: {}".format(tf.test.gpu_device_name()))
            else:
                print_verbose("GPU not detected, please ensure Drivers/CUDA/cuDNN/tf-gpu are set up properly")
                print_verbose("Continuing with CPU")
                args.gpu = False

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
        print("Training complete, models available in ./saved_models/")
    elif args.command == "squig":
        squig_pipeline(args)
    else:
        print_err("command: {} not recognised".format(args.command))
        parser.print_help(sys.stderr)
        sys.exit(1)

    # done!

    # # TODO: sub-module


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
    # make this dynamic
    # barcode_out = {0: "bc_1",
    #                1: "bc_2",
    #                2: "bc_3",
    #                3: "bc_4",
    #                None: "unknown"
    #                }
    # if someone does more than 50 samples I will skull a beer cause that's awesome!
    # left here to remind me when I have to make the update
    barcode_out = {i: "bc_{}".format(i+1) for i in range(0, 50)}
    barcode_out[None] = "unknown"
    labels = []
    images = []
    fast5s = {}
    stats = ""
    seg_dic = {}
    if args.squiggle:
        squig_file = args.squiggle
        with open(squig_file, 'a') as f:
            f.write("{}\t{}\n".format("ReadID", "signal_pA"))
    else:
        squig_file = ''
    # TODO: sub-module
    if args.segment:
        seg_file = args.segment
        with open(seg_file, 'a') as f:
            f.write("{}\t{}\t{}\n".format("ReadID", "start", "stop"))
    else:
        seg_file = ""
    # make this dynamic for number of barcodes
    print("{}\t{}\t{}\t{}\t{}".format("fast5", "ReadID", "Barcode", "Confidence Interval", "\t".join(["P_bc_{}".format(i) for i in range(1,args.Number+1)])))
    # for file in input...
    # TODO: sub-module
    fnames = []
    for path in args.path:
        fnames += list(sorted(map(str, Path(path).rglob('*.fast5'))))
        
    p = Pool(args.threads, maxtasksperchild=1)
    _args = [(f5, window, squig_file, seg_file, args.test) for f5 in fnames]
    for fi, (fast5_file, seg_signal) in enumerate(p.imap_unordered(worker, _args), 1):
        # get fname
        fast5 = os.path.basename(fast5_file)
        sys.stderr.write("%s / %s %s\n"%(fi, len(fnames), fast5_file))
        #everthing below this, send off in batches of N=args.batch_size
        # The signal extraction and segmentation can happen in the first step
        # read fast5 files
        for sig_count, readID in enumerate(seg_signal, 1):
            # convert - it may be even better to convert to image within subprocess
            img = convert_to_image(np.array(seg_signal[readID], dtype=float))
            labels.append(readID)
            images.append(img)
            fast5s[readID] = fast5
            # TODO: sub-module
            if len(labels) >= args.batch_size:
                C = classify(model, labels, np.array(images), False, args.threshold)
                # save to output
                for readID, out, c, P in C:
                    prob = [round(float(i), 6) for i in P]
                    cm = round(float(c), 4)
                    if args.verbose:
                        print_verbose("cm is: {}".format(cm))
                    # make this dynamic
                    print("{}\t{}\t{}\t{}\t{}".format(fast5s[readID], readID, barcode_out[out], cm, "\t".join(["{:.5f}".format(prob[i]) for i in range(0,len(prob))])))
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
        # Make this dynamic
        print("{}\t{}\t{}\t{}\t{}".format(fast5s[readID], readID, barcode_out[out], cm, "\t".join(["{:.5f}".format(prob[i]) for i in range(0,len(prob))])))
    images = []
    fast5s = {}

    # final report/stats
    # print stats
    return

def worker(args):
    fast5_file, window, squig_file, seg_file, test = args
    seg_signal = get_multi_fast5_signal(fast5_file, window, squig_file, seg_file, test=test)
    return fast5_file, seg_signal
    
if __name__ == '__main__':
    main()
