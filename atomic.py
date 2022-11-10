import json
import csv
import random
from collections import defaultdict

def node_iterator(filename):
    with open(filename) as f:
        rdr = csv.reader(f, delimiter="\t", quotechar='"')

        # get line count first
        head = ''
        lastHead = head
        subgraph = defaultdict(list)
        for triplet in rdr:
            head = triplet[0]
            if head == lastHead:
                if triplet[2] == 'none':
                    continue
                subgraph[head].append(triplet[1:])
            else:
                if len(subgraph) > 0:
                    yield subgraph
                subgraph = defaultdict(list)
            lastHead = head
            
