import json
import csv
import random

def node_iterator(filename):
    with open(filename) as f:
        rdr = csv.reader(f, delimiter="\t", quotechar='"')
        head = ''
        lastHead = head
        subgraph = []
        for triplet in rdr:
            head = triplet[0]
            if head == lastHead:
                if triplet[2] == 'none':
                    continue
                subgraph.append(triplet)
            else:
                if len(subgraph) > 0:
                    yield subgraph
                subgraph = []
            lastHead = head
            
