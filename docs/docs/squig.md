# Extra Squiggle tools

These tools are extensions or modified versions of [SquiggleKit](https://github.com/Psy-Fer/SquiggleKit).


### Extract squggles from fast5 files

Equivalent to [SquigglePull](https://github.com/Psy-Fer/SquiggleKit/blob/master/SquigglePull.py)

    python3 deeplexicon.py squig -p /path/to/fast5s/ > squiggles.tsv

### Segment squiggles from squiggle.tsv

    python3 deeplexicon.py squig -s squiggle.tsv > segments.tsv


<!-- TODO: Breakdown of extra options, especially dRNA_segmenter -->
