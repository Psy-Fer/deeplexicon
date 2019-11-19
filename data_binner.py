import os
import sys
import argparse
'''

    James M. Ferguson (j.ferguson@garvan.org.au)
    Genomic Technologies
    Garvan Institute
    Copyright 2019

    data_binner.py

    This script is used to take readIDs that were cut from an alignemt file and
    put into a flat file. It creates dictionaries for the list of input flat files,
    then splits those into squiggle and dmux files which have the segments.

    ----------------------------------------------------------------------------
    version 0.0 - initial



    TODO:
        -

    ----------------------------------------------------------------------------
    MIT License

    Copyright (c) 2019 James Ferguson

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


def main():
    '''
    do the thing
    '''
    parser = MyParser(
        description="data_binner.py - takes binned readIDs and splits signal and dmux data")
    #group = parser.add_mutually_exclusive_group()
    parser.add_argument("-s", "--squiggle",
                        help="Squiggle file")
    parser.add_argument("-d", "--dmux",
                        help="dmux file")
    parser.add_argument("-b", "--binned",
                        help="binned files as comma list")

    args = parser.parse_args()

    # print help if no arguments given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    bins = args.binned.split(',')

    bin_dic = {}

    for bin in bins:
        bin_dic[bin] = set()
        with open(bin, 'rt') as f:
            for l in f:
                l = l.strip('\n')
                l = l.split('\t')
                bin_dic[bin].add(l[0])

    # split signals no header
    squig_dic = {}
    for bin in bins:
        squig_dic[bin] = {}
    squig_dic['nobin'] = {}
    C = 0
    T = 0
    with open(args.squiggle, 'rt') as f:
        for l in f:
            C += 1
            T += 1
            found = False
            readID = l.split('\t')[0]
            for bin in bins:
                if readID in bin_dic[bin]:
                    squig_dic[bin][readID] = l
                    found = True
            if not found:
                squig_dic['nobin'][readID] = l

            if C > 1000:
                sys.stderr.write('info: progress {}.\n'.format(T))
                for bin in bins:
                    with open(bin+"_squiggle.tsv", 'a') as w:
                        for readID in squig_dic[bin]:
                            w.write(squig_dic[bin][readID])
                with open('nobin_squiggle.tsv', 'a') as w:
                    for readID in squig_dic['nobin']:
                        w.write(squig_dic['nobin'][readID])

                squig_dic = {}
                for bin in bins:
                    squig_dic[bin] = {}
                squig_dic['nobin'] = {}
                C = 0

        for bin in bins:
            with open(bin+"_squiggle.tsv", 'a') as w:
                for readID in squig_dic[bin]:
                    w.write(squig_dic[bin][readID])
        with open('nobin_squiggle.tsv', 'a') as w:
            for readID in squig_dic['nobin']:
                w.write(squig_dic['nobin'][readID])

    # nuke dic
    squig_dic = {}

    # split dmux with header
    dmux_dic = {}
    for bin in bins:
        dmux_dic[bin] = {}
    dmux_dic['nobin'] = {}
    C = 0
    T = 0
    head = True
    with open(args.dmux, 'rt') as f:
        for l in f:
            if head:
                head = False
                header = l
                for bin in bins:
                    with open(bin+"_dmux.tsv", 'a') as w:
                        w.write(header)
                with open('nobin_dmux.tsv', 'a') as w:
                    w.write(header)
                continue
            C += 1
            T += 1
            found = False
            readID = l.split('\t')[1]
            for bin in bins:
                if readID in bin_dic[bin]:
                    dmux_dic[bin][readID] = l
                    found = True
            if not found:
                dmux_dic['nobin'][readID] = l

            if C > 1000:
                sys.stderr.write('info: progress {}.\n'.format(T))
                for bin in bins:
                    with open(bin+"_dmux.tsv", 'a') as w:
                        for readID in dmux_dic[bin]:
                            w.write(dmux_dic[bin][readID])
                with open('nobin_dmux.tsv', 'a') as w:
                    for readID in dmux_dic['nobin']:
                        w.write(dmux_dic['nobin'][readID])

                dmux_dic = {}
                for bin in bins:
                    dmux_dic[bin] = {}
                dmux_dic['nobin'] = {}
                C = 0

        for bin in bins:
            with open(bin+"_dmux.tsv", 'a') as w:
                for readID in dmux_dic[bin]:
                    w.write(dmux_dic[bin][readID])
        with open('nobin_dmux.tsv', 'a') as w:
            for readID in dmux_dic['nobin']:
                w.write(dmux_dic['nobin'][readID])



if __name__ == '__main__':
    main()
