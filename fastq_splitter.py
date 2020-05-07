import sys, os
import argparse

'''

    James M. Ferguson (j.ferguson@garvan.org.au)
    Genomic Technologies
    Garvan Institute
    Copyright 2019

    Split a single fastq file into individual barcode fastqs

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
        description="fastq splitter - run on a deeplexicon output tsv")
    #group = parser.add_mutually_exclusive_group()
    parser.add_argument("-d", "--dmux",
                        help="deeplexicon output tsv file")
    parser.add_argument("-q", "--fastq",
                        help="single combined fastq file")
    parser.add_argument("-o", "--output",
                        help="output path")
    parser.add_argument("-s", "--sample", default="dmux_",
                        help="sample name to append to file names")


    args = parser.parse_args()

    # print help if no arguments given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    read_bcs, bc_set = get_reads(args.dmux)

    split_fastq(read_bcs, bc_set, args.fastq, args.output, args.sample)


def get_reads(filename):
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


def split_fastq(read_bcs, bc_set, fastq, output, sample):
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


if __name__ == '__main__':
    main()
