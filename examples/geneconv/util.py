"""
"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_equal


def ad_hoc_fasta_reader(fin):
    name_seq_pairs = []
    while True:

        # read the name
        line = fin.readline().strip()
        if not line:
            return name_seq_pairs
        assert_equal(line[0], '>')
        name = line[1:].strip()

        # read the single line sequence
        line = fin.readline().strip()
        seq = line
        unrecognized = set(line) - set('ACGT')
        if unrecognized:
            raise Exception('unrecognized nucleotides: ' + str(unrecognized))

        name_seq_pairs.append((name, seq))
