import numpy as np
from rle import *

class Node:
    
    def __init__(self, symbol, prob, left=None, right=None):

        self.symbol = symbol

        self.prob = prob

        self.left = left

        self.right = right

        self.code = ''

# Returns a dictionary holding the estimated probability of each symbol
def get_probabilities(run_symbols):

    # Get number of unique symbols
    unique_symbols = set(run_symbols)

    # Initialize a dictionary to hold each symbol's probabilities
    symbol_probs = dict.fromkeys(unique_symbols,0)

    # Get total number of symbols
    num_of_symbols = len(run_symbols)

    # For each symbol
    for s in unique_symbols:

        # Get its probability (Occurances/Total Symbols)
        symbol_probs[s] = run_symbols.count(s) / num_of_symbols

    return symbol_probs

def get_codes(node, codes, current_code=''):
    code = current_code + node.code

    if node.left:
        get_codes(node.left, codes, code)
    
    if node.right:
        get_codes(node.right, codes, code)
    
    if not node.left and not node.right:
        codes[node.symbol] = code

    return codes

def huff(run_symbols):

    frame_stream = ''
    frame_symbol_prob = get_probabilities(run_symbols)

    unique_symbols = set(run_symbols)
    codes = dict.fromkeys(unique_symbols,'')

    nodes = []

    for s in unique_symbols:
        node = Node(s, frame_symbol_prob[s])
        nodes.append(node)

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda n: n.prob)
        left = nodes[0]
        right = nodes[1]

        left.code = '0'
        right.code = '1'

        node = Node('',left.prob+right.prob, left, right)

        del nodes[0:2]
        
        nodes.append(node)

    # Only one node is left, so iterate through the tree to get the codes
    codes = get_codes(nodes[0], codes)

    # Then build the bit stream using the generated codes
    for s in run_symbols:
        frame_stream += codes[s]

    return frame_stream, frame_symbol_prob


x = [0,0,1,0,0,0,0,3,0,0,1,0,2,0,2,0,2]
symb_index = RLE0(x)

get_probabilities(symb_index)

frame_stream, frame_symbol_prob = huff(symb_index)

print(symb_index)

print(frame_stream)