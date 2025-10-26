"""
Hashcat Rule Extractor:
Identifies single and chained Hashcat rules that transform words 
from a base wordlist into passwords found in a target password list. 
It utilizes multiprocessing for efficient analysis and outputs rules 
sorted by frequency.
"""

import os
import argparse
from tqdm import tqdm
from collections import Counter
import string
from multiprocessing import Pool, cpu_count
# WARNING: These Cython imports are for reference only. 
# This script is a Python-only version for simplicity.
# from cython.cimports.libc.stdlib import strtol
# from cython.cimports.libc.string import strlen

# --- Base36 Conversion (Python Fallback) ---
def i36(s):
    try:
        # If 's' is not a string (e.g., in rule generator for single char)
        if isinstance(s, bytes):
            s = s.decode('latin-1')
        return int(s, 36)
    except (ValueError, IndexError, TypeError):
        return -1

# The C-optimized function is removed from the Python file to prevent runtime errors.
# cpdef int c_i36(bytes s): ...

# --- Rule Functions (FUNCTS) ---
FUNCTS = {}
FUNCTS[':'] = lambda x, i: x
FUNCTS['l'] = lambda x, i: x.lower()
FUNCTS['u'] = lambda x, i: x.upper()
FUNCTS['c'] = lambda x, i: x.capitalize()
def C(x, i):
    return x[0].lower() + x[1:].upper() if len(x) > 0 else ""
FUNCTS['C'] = C
FUNCTS['t'] = lambda x, i: x.swapcase()
def T(x, i):
    number = i36(i)
    if number < 0 or number >= len(x): return None
    return ''.join((x[:number], x[number].swapcase(), x[number + 1:]))
FUNCTS['T'] = T
FUNCTS['r'] = lambda x, i: x[::-1]
FUNCTS['d'] = lambda x, i: x+x
def p(x, i):
    number = i36(i)
    if number < 0: return None
    # x*0 returns '' in Python, but Hashcat p0 returns x (i.e., x*(0+1))
    return x*(number+1) 
FUNCTS['p'] = p
FUNCTS['f'] = lambda x, i: x+x[::-1]
FUNCTS['{'] = lambda x, i: x[1:]+x[0] if len(x) > 0 else x
FUNCTS['}'] = lambda x, i: x[-1]+x[:-1] if len(x) > 0 else x
# Corrected function names for '$' and '^'
def rule_append_char(x, i): return x+i
def rule_prepend_char(x, i): return i+x
FUNCTS['$'] = rule_append_char 
FUNCTS['^'] = rule_prepend_char
FUNCTS['['] = lambda x, i: x[1:]
FUNCTS[']'] = lambda x, i: x[:-1]
def D(x, i):
    number = i36(i)
    if number < 0 or number >= len(x): return None
    return x[:number] + x[number+1:]
FUNCTS['D'] = D
def x_ext(x, i):
    if len(i) != 2: return None
    n = i36(i[0])
    m = i36(i[1])
    if n < 0 or m < 0: return None
    return x[n:n+m]
FUNCTS['x'] = x_ext
def O(x, i):
    if len(i) != 2: return None
    n = i36(i[0])
    m = i36(i[1])
    if n < 0 or m < 0: return None
    return x[:n] + x[n+m:]
FUNCTS['O'] = O
def i_ins(x, i):
    if len(i) != 2: return None
    n = i36(i[0])
    char = i[1]
    if n < 0 or n > len(x): return None
    return x[:n] + char + x[n:]
FUNCTS['i'] = i_ins
def o_ovw(x, i):
    if len(i) != 2: return None
    n = i36(i[0])
    char = i[1]
    if n < 0 or n >= len(x): return None
    return x[:n] + char + x[n+1:]
FUNCTS['o'] = o_ovw
def apostrophe(x, i):
    n = i36(i)
    if n < 0 or n > len(x): return None
    return x[:n]
FUNCTS["'"] = apostrophe
def s(x, i):
    # Hashcat s<old><new> replaces ALL occurrences.
    if len(i) == 2:
        return x.replace(i[0], i[1])
    return None
FUNCTS['s'] = s
FUNCTS['@'] = lambda x, i: x.replace(i, '')
def z(x, i):
    number = i36(i)
    return x[0]*number + x if len(x) > 0 and number >= 0 else x
FUNCTS['z'] = z
def Z(x, i):
    number = i36(i)
    return x+x[-1]*number if len(x) > 0 and number >= 0 else x
FUNCTS['Z'] = Z
FUNCTS['q'] = lambda x, i: ''.join([a*2 for a in x])
def k(x,i):
    if len(x) < 2: return x
    return x[1]+x[0]+x[2:]
FUNCTS['k'] = k
def K(x,i):
    if len(x) < 2: return x
    return x[:-2]+x[-1]+x[-2]
FUNCTS['K'] = K
def star(x, i):
    if len(i) != 2: return None
    n = i36(i[0])
    m = i36(i[1])
    if n < 0 or m < 0 or n >= len(x) or m >= len(x): return None
    chars = list(x)
    chars[n], chars[m] = chars[m], chars[n]
    return ''.join(chars)
FUNCTS['*'] = star
def L(x, i):
    n = i36(i)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    shifted_val = char_val << 1
    return x[:n] + chr(shifted_val) + x[n+1:]
FUNCTS['L'] = L
def R(x, i):
    n = i36(i)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    shifted_val = char_val >> 1
    return x[:n] + chr(shifted_val) + x[n+1:]
FUNCTS['R'] = R
def plus(x, i):
    n = i36(i)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    return x[:n] + chr(char_val + 1) + x[n+1:]
FUNCTS['+'] = plus
def minus(x, i):
    n = i36(i)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    return x[:n] + chr(char_val - 1) + x[n+1:]
FUNCTS['-'] = minus
def dot(x, i):
    n = i36(i)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    return x[:n] + chr(char_val + 1) + x[n+1:]
FUNCTS['.'] = dot
def comma(x, i):
    n = i36(i)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    return x[:n] + chr(char_val - 1) + x[n+1:]
FUNCTS[','] = comma
def y(x, i):
    n = i36(i)
    if n < 0 or n > len(x): return None
    return x[:n] + x
FUNCTS['y'] = y
def Y(x, i):
    n = i36(i)
    if n < 0 or n > len(x): return None
    return x + x[-n:]
FUNCTS['Y'] = Y
def E(x, i):
    return string.capwords(x, ' ')
FUNCTS['E'] = E
def e(x, i):
    if len(i) != 1: return None
    return string.capwords(x, i)
FUNCTS['e'] = e
def three(x, i):
    if len(i) != 2: return None
    n = i36(i[0])
    separator = i[1]
    parts = x.split(separator)
    if n < 0 or n >= len(parts): return None
    parts[n] = parts[n].swapcase()
    return separator.join(parts)
FUNCTS['3'] = three


class RuleEngine(object):
    """
    Applies Hashcat rules to a word, managing both single and chained rules.
    """
    def apply_rule(self, rule_str, word):
        """Applies a single Hashcat rule (or chain) to a word."""
        parsed_rule = self._parse_rule(rule_str)
        if parsed_rule is None:
            return None
            
        current_word = word
        for function_id, arg in parsed_rule:
            if current_word is None:
                return None
            
            try:
                # Use the function mapped in the global FUNCTS dictionary
                # Pass original word if it's the first step in the chain 
                # (though this implementation passes only the arg)
                current_word = FUNCTS[function_id](current_word, arg)
            except (IndexError, ValueError, TypeError, KeyError):
                return None
        
        # Hashcat rules must not result in an empty string
        return current_word if current_word else None

    def _parse_rule(self, rule_str):
        """Parses a rule string into an ordered list of (function_char, argument) tuples."""
        rules = []
        i = 0
        while i < len(rule_str):
            char = rule_str[i]
            
            # The parsing logic needs to cover ALL characters, including those in chains.
            # Using the original parser logic, adapted to the fixed function list:

            # 1-char rule with no argument
            if char in "ludtrf{}qkK[]CcE":
                rules.append((char, ""))
                i += 1
            
            # 2-char rule with 1 argument (char or base36 position)
            elif char in "$^@TD'LRTZz+-.,pyY":
                if i + 1 < len(rule_str):
                    rules.append((char, rule_str[i+1]))
                    i += 2
                else:
                    return None
            
            # 3-char rule with 2 arguments (base36 + char/base36)
            elif char in "xO*sioe":
                if i + 2 <= len(rule_str):
                    rules.append((char, rule_str[i+1:i+3]))
                    i += 3
                else:
                    return None
            
            # 4-char rule (3<pos><sep>)
            elif char == '3':
                if i + 2 <= len(rule_str):
                    rules.append((char, rule_str[i+1:i+3]))
                    i += 3
                else:
                    return None
            
            else:
                return None # Invalid operator
        return rules

# --- Multiprocessing Globals and Workers ---

# Global variables for the RuleEngine object in each worker process
rule_engine_instance = None
target_set_global = None
rules_global = None
chain_depths_global = None

def init_worker(rules, target_set, chain_depths):
    """Initializer function for the multiprocessing Pool."""
    global rule_engine_instance
    global target_set_global
    global rules_global
    global chain_depths_global
    rule_engine_instance = RuleEngine()
    target_set_global = target_set
    rules_global = rules
    chain_depths_global = chain_depths

def generate_chains(rules, depth):
    """
    Generator to recursively create rule chains of a specified depth.
    Since chains of length 1 are handled explicitly, this targets depth >= 2.
    """
    if depth == 2:
        for r1 in rules:
            for r2 in rules:
                yield f"{r1}{r2}"
    elif depth == 3:
        for r1 in rules:
            for r2 in rules:
                for r3 in rules:
                    yield f"{r1}{r2}{r3}"
    # Higher depths are possible but become prohibitively slow (N^depth)

def process_word(word):
    """
    The main worker function executed by each process, checking chains.
    """
    global rule_engine_instance
    global target_set_global
    global rules_global
    global chain_depths_global
    
    found_rules_local = Counter()
    
    # Iterate through each required chain depth
    for depth in chain_depths_global:
        
        # Depth 1: Single rules
        if depth == 1:
            for rule in rules_global:
                transformed_word = rule_engine_instance.apply_rule(rule, word)
                if transformed_word and transformed_word != word and transformed_word in target_set_global:
                    found_rules_local[rule] += 1
            
        # Depth 2 & 3: Chains
        elif depth > 1 and depth <= 3:
            # Note: This is an exhaustive check (O(N^depth)).
            # Optimizations would involve checking which base rules actually produce an intermediate word.
            
            # We skip chains that were already found as part of shorter chains.
            # Example: If a rule is 'l$' and another is 'lu', skip 'l' and 'u' if the depth is 2. 
            # This is complex to manage efficiently across all workers. For simplicity,
            # we rely on the Counter storing all hits and ignore the performance hit 
            # of checking 'l' + 'u' even if 'lu' was already checked in depth 2.
            
            for rule_chain in generate_chains(rules_global, depth):
                transformed_word = rule_engine_instance.apply_rule(rule_chain, word)
                if transformed_word and transformed_word != word and transformed_word in target_set_global:
                    found_rules_local[rule_chain] += 1
                        
    return found_rules_local

# --- Data Handling and Rule Generation (Kept mostly as original) ---

def load_data(filename):
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist.")
        return None
    try:
        # Use a more robust encoding strategy
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"An error occurred while loading the file '{filename}': {e}")
        return None

def encode_non_ascii_to_hex(rule):
    try:
        rule.encode('ascii')
        return rule
    except UnicodeEncodeError:
        hex_encoded_rule = ""
        for char in rule:
            try:
                char.encode('ascii')
                hex_encoded_rule += char
            except UnicodeEncodeError:
                hex_encoded_rule += "".join([f"\\x{ord(byte):02x}" for byte in char.encode('latin-1')])
        return hex_encoded_rule

def generate_simple_rules():
    """Generates the core set of single Hashcat rules to test."""
    
    simple_rules = set()
    # 1. Single character rules
    simple_rules.update([':', 'l', 'u', 'c', 'C', 't', 'r', 'f', 'd', 'q', '{', '}', '[', ']', 'k', 'K', 'E'])

    # 2. Simple duplications (p) and truncations ([], [])
    for i in range(1, 4): simple_rules.add('p' + str(i))
    for i in range(1, 4): simple_rules.add(']' * i)
    for i in range(1, 4): simple_rules.add('[' * i)

    # 3. Append ($) and Prepend (^) rules
    CHARS_TO_APPEND = string.digits + string.ascii_letters + "!@#$%^&*()_+-="
    for char in CHARS_TO_APPEND:
        simple_rules.add(f"${char}")
        simple_rules.add(f"^{char}")

    # 4. Positional rules 
    POSITIONS = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f'] # 0-15
    POSITIONAL_OPS = ['T', 'D', "'", 'L', 'R', '+', '-', '.', ',', 'y', 'Y', 'z', 'Z']
    for pos in POSITIONS:
        for op in POSITIONAL_OPS:
            simple_rules.add(f"{op}{pos}")

    # 5. Substitution (s) and Deletion (@)
    SUBSTITUTIONS = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7'}
    for k, v in SUBSTITUTIONS.items():
        simple_rules.add(f"s{k}{v}")
        simple_rules.add(f"@{k}")

    # 6. Swap (*) rules (limited to common positions)
    SWAP_POS = ['0', '1', '2', '3']
    for pos_a in SWAP_POS:
        for pos_b in SWAP_POS:
            if pos_a != pos_b:
                simple_rules.add(f"*{pos_a}{pos_b}")

    # 7. Insert (i) and Overwrite (o) rules (limited tests)
    for pos in SWAP_POS:
        for char in ['1', '!', 'a', 'A']:
             simple_rules.add(f"i{pos}{char}")
             simple_rules.add(f"o{pos}{char}")

    # 8. Split/Capitalize rules
    simple_rules.add(f"e ")
    simple_rules.add(f"30 ")

    return simple_rules

# --- Main Logic and Argument Parsing ---

def main():
    parser = argparse.ArgumentParser(description='Hashcat rule extractor based on example passwords.')
    parser.add_argument('-b', '--base-words', required=True, help='Path to the file with base words.')
    parser.add_argument('-t', '--target-passwords', required=True, help='Path to the file with target passwords.')
    parser.add_argument('-o', '--output-file', default='extracted_rules.rule', help='Path to the output file for the rules. Default: extracted_rules.rule')
    parser.add_argument('-c', '--cores', type=int, default=cpu_count(), help=f'Number of CPU cores to use. Default: {cpu_count()}')
    parser.add_argument('--chain-range', type=str, default='1-1', 
                        help='Range of rule chain depths to test (e.g., "1-2" for depth 1 and 2, "3-3" for depth 3 only). Max depth is 3. Default: 1-1')
    
    args = parser.parse_args()

    # Parse chain range
    try:
        min_depth, max_depth = map(int, args.chain_range.split('-'))
        if not (1 <= min_depth <= 3 and 1 <= max_depth <= 3 and min_depth <= max_depth):
             raise ValueError("Depth must be between 1 and 3, and min_depth must be <= max_depth.")
        chain_depths = list(range(min_depth, max_depth + 1))
    except ValueError as e:
        print(f"Error parsing --chain-range: {e}")
        return

    print("--- Hashcat Rule Extractor (Flexible Chain Depth) ---")
    print(f"Base Words: '{args.base_words}' | Target Passwords: '{args.target_passwords}'")
    print(f"Chain Depths: {', '.join(map(str, chain_depths))}")
    print(f"Cores: {args.cores}")
    print("-----------------------------------------------------")

    base_words = load_data(args.base_words)
    target_passwords = load_data(args.target_passwords)
    
    if base_words is None or target_passwords is None:
        return
        
    target_set = set(target_passwords)
    
    # 1. Generate Rules
    simple_rules_list = list(generate_simple_rules())
    print(f"Generated a core set of {len(simple_rules_list)} single rules for testing.")
    print(f"Analyzing {len(base_words)} base words against {len(target_set)} unique target passwords.")

    # 2. Prepare Multiprocessing
    initializer_args = (simple_rules_list, target_set, chain_depths)
    found_rules = Counter()
    
    num_processes = min(args.cores, cpu_count())
    
    print(f"Starting rule extraction using {num_processes} processes...")
    
    try:
        with Pool(processes=num_processes, initializer=init_worker, initargs=initializer_args) as p:
            results = p.imap_unordered(process_word, base_words, chunksize=100)
            
            for result in tqdm(results, total=len(base_words), desc="Processing words"):
                found_rules.update(result)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving partial results...")
    
    print("-----------------------------------------")
    if not found_rules:
        print("No matching rules found.")
        return
        
    print(f"Found {len(found_rules)} unique rules.")

    print("\n--- Top 10 Most Frequent Rules ---")
    for rule, count in found_rules.most_common(10):
        # Limit the displayed rule length for clean output
        print(f"  {rule[:20]:<20} (Occurrences: {count})")
    
    print("-----------------------------------------")

    # 3. Output Results
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for rule, count in found_rules.most_common():
                hex_rule = encode_non_ascii_to_hex(rule)
                f.write(f"{hex_rule}\n")
        print(f"All rules, sorted by frequency, were successfully saved to '{args.output_file}'.")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
        
if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()

