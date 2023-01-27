import numpy as np

def RLE(symb_index):

    # Initialize the count 
    count = 1

    # Set the first previous symbol as the first in the array
    s_old = symb_index[0]

    # A list holding the tuples (symbol, length)
    run_symbols = []

    # For each symbol in the symbol indices
    for i in range(1,len(symb_index)):

        # Get the new symbol
        s_new = symb_index[i]

        # Compare the old and new symbol.
        # If they are the same, increase the count
        if s_new == s_old:
            count += 1
        # Else, append the symbol and its length as a tuple and reset the count
        else:
            run_symbols.append((s_old, count))
            count = 1
        
        # Set the new S as the old S
        s_old = s_new

    # Append the last run of symbols
    run_symbols.append((s_old, count))

    return run_symbols


def IRLE(run_symbols, K):

    # Initialize an array of symbols
    symb_index = np.zeros(K)

    # Initialize an index showing the index to currently add to in the symb_index array
    idx = 0

    # For each pair of (symbol, length)
    for symbol_pair in run_symbols:

        # Get the symbol and its run length
        symbol = symbol_pair[0]
        length = symbol_pair[1]

        # Add the appropriate number of symbols to the symb_idx array
        symb_index[idx:idx+length] = symbol

        # Update the current index
        idx += length

    return symb_index

def RLE0(symb_index):

    # Initialize the count 
    count = 0

    # Set the first previous symbol as the first in the array
    #s_old = symb_index[0]

    # A list holding the tuples (quant_symbol, following_zeros)
    run_symbols = []

    # For each symbol in the symbol indices
    for i in range(len(symb_index)):

        # Get the new symbol
        s = symb_index[i]

        # Check if it is a following zero (and not the last element)
        # (If it is the last element, a symbol in the form of (0, N) will be added, where N the N last 0s)
        if s == 0 and i != (len(symb_index)-1):
            count += 1
        # Else, append the symbol and its length as a tuple and reset the count
        else:
            run_symbols.append((s, count))
            count = 0


    return run_symbols


def IRLE0(run_symbols, K):

    # Initialize an array of symbols
    symb_index = np.zeros(K)

    # Initialize an index showing the index to currently add to in the symb_index array
    idx = 0

    # For each pair of (symbol, length)
    for symbol_pair in run_symbols:

        # Get the symbol and its run length
        symbol = symbol_pair[0]
        following_zeros = symbol_pair[1]

        # Add the appropriate number of symbols to the symb_idx array
        symb_index[idx:idx+following_zeros] = 0
        symb_index[idx+following_zeros] = symbol

        # Update the current index
        idx += following_zeros + 1

    return symb_index

x = [0,0,1,0,0,0,0,3,0,2,0,0,2,0,0]

symb_index = RLE0(x)

print(symb_index)

xhat = IRLE0(symb_index, len(x))

print(xhat)