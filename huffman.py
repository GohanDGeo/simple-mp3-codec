# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# huffman.py
#
# This file contains function which calculate the huffman code for a series of symbols.
# It also contains functions that reverse this process, producing a series of symbols from the huffman code.

# A Node class, used to build the huffman tree.
# Each node has a reference to its left and right connected neighbors,
# as well as information about its symbol (only for leaf nodes) and its code (0/1)
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

# A recursive function, which calculates the code for each node, and
# for each leaf node, appends its resulting code to the dictionary @codes.
def get_codes(node, codes, current_code=''):

    # Set the current code to the code accumulated so far + the node's code
    code = current_code + node.code

    # If the node has a left neighbor, recursively call the function
    if node.left:
        get_codes(node.left, codes, code)
    
    # Same for the right node
    if node.right:
        get_codes(node.right, codes, code)
    
    # If the node has no neighbors, it is a leaf,
    # therefore its code is appended to the dictionary, which contains
    # the huffman codes for the symbols
    if not node.left and not node.right:
        codes[node.symbol] = code

    return codes

# The main function for generating the huffman code
def huff(run_symbols):

    # Initialize an empty string for the bitstream
    frame_stream = ''
    # Get the probabilities for each symbol
    frame_symbol_prob = get_probabilities(run_symbols)

    # Get the set of unique symbols
    unique_symbols = set(run_symbols)
    # Initialize an empty dictionary, having each unique symbol as a key
    # and its huffman code as the value (initialized to an empty string)
    codes = dict.fromkeys(unique_symbols,'')

    # Initialize a list of nodes that have to be merged in order to create the huffman tree
    nodes = []

    # First, create a node for each unique symbol, and append it to the list
    for s in unique_symbols:
        node = Node(s, frame_symbol_prob[s])
        nodes.append(node)

    # Then, while the list has more than 1 element
    while len(nodes) > 1:
        # Get the 2 nodes with the lowest probabilities
        nodes = sorted(nodes, key=lambda n: n.prob)
        left = nodes[0]
        right = nodes[1]

        # Convention that the left neighbor has the code 0
        # and the right the code 1
        left.code = '0'
        right.code = '1'

        # Create a new node with the sum of the probabilities of the two nodes,
        # and that has the 2 nodes as neighbors
        node = Node('',left.prob+right.prob, left, right)

        # Delete the two nodes from the list, as they have been merged
        del nodes[0:2]
        
        # And append the newly created node to the list
        nodes.append(node)

    # Only one node is left, so iterate through the tree to get the codes
    codes = get_codes(nodes[0], codes)

    # Then build the bit stream using the generated codes
    # For each symbol encountered in the @run_symbols, get its code 
    # and append it to the bitstream string
    for s in run_symbols:
        frame_stream += codes[s]

    return frame_stream, frame_symbol_prob

# This function does the inverse of the huffman code: 
# Generates a run of symbols, given a bitstream (generated using huffman code)
# and the probabilities for each symbol (along with the symbols, in the form of a dict)
def ihuff(frame_stream, frame_symbol_prob):

    # Initialize an empty list for the symbols
    run_symbols = []

    # Get the unique symbols, which are the keys of the probabilities dict
    unique_symbols = frame_symbol_prob.keys()

    # And now, the same process as above is done.
    # Since only the probabilities of the symbols is given,
    # in order to decrypt the huffman code, the huffman tree has to 
    # be generated again, following the convention that the left neighbor gets
    # code 0, and the right 1
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

    # After the huffman tree is generated again, the decryption begins

    # Get a reference to the root node
    root = nodes[0]

    # And a reference to the current node
    curr_node = root

    # The decryption of each symbol begins from the root.
    # Each bit is read one-by-one, and the huffman tree is navigated,
    # going left each time a 0 is encoutnered, and right each time a 1 is
    # encountered. If the node we navigate to has no neighbors, it is a leaf
    # node, so the bitstream read thus far, corresponds to the symbol of
    # this leaf node, and the symbol is added to the @run_symbols
    # This process is repeated until the whole bitstream has been read,
    # resetting to the root each time a leaf node is found.
    for b in frame_stream:
        if b == '0':
            curr_node = curr_node.left
        elif b == '1':
            curr_node = curr_node.right
        
        if not curr_node.right and not curr_node.left:
            run_symbols.append(curr_node.symbol)
            curr_node = root
    
    return run_symbols