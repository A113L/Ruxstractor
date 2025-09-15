# distutils: extra_compile_args=-O3
# distutils: language=c
# distutils: extra_compile_args=-O3
# distutils: language=c
import os
import argparse
import sys
import string
import itertools
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count
from cython.cimports.libc.stdlib import strtol
from cython.cimports.libc.string import strlen

def i36(s):
    # This function is now just a Python wrapper, the C-optimized version is below.
    try:
        return int(s, 36)
    except (ValueError, IndexError):
        return -1

cpdef int c_i36(bytes s):
    cdef long val = 0
    cdef char* endptr
    val = strtol(s, &endptr, 36)
    if endptr[0] != b'\0':
        return -1
    return val

FUNCTS = {}
FUNCTS[':'] = lambda x, i: x
FUNCTS['l'] = lambda x, i: x.lower()
FUNCTS['u'] = lambda x, i: x.upper()
FUNCTS['c'] = lambda x, i: x.capitalize()
def C(x, i):
    if len(x) > 0: return x[0].lower() + x[1:].upper()
    return ""
FUNCTS['C'] = C
FUNCTS['t'] = lambda x, i: x.swapcase()
def T(x, i):
    number = i36(i)
    if number < 0:
        number += len(x)
    if number < 0 or number >= len(x):
        return None
    return ''.join((x[:number], x[number].swapcase(), x[number + 1:]))
FUNCTS['T'] = T
FUNCTS['r'] = lambda x, i: x[::-1]
FUNCTS['d'] = lambda x, i: x+x
def p(x, i):
    number = i36(i)
    if number < 0: return None
    return x*(number+1)
FUNCTS['p'] = p
FUNCTS['f'] = lambda x, i: x+x[::-1]
FUNCTS['{'] = lambda x, i: x[1:]+x[0] if len(x) > 0 else x
FUNCTS['}'] = lambda x, i: x[-1]+x[:-1] if len(x) > 0 else x
FUNCTS['$'] = lambda x, i: x+i
FUNCTS['^'] = lambda x, i: i+x
FUNCTS['['] = lambda x, i: x[1:]
FUNCTS[']'] = lambda x, i: x[:-1]
def D(x, i):
    number = i36(i)
    if number < 0:
        number += len(x)
    if number < 0 or number >= len(x):
        return None
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
    if n < 0:
        n += len(x) + 1
    if n < 0 or n > len(x):
        return None
    return x[:n] + char + x[n:]
FUNCTS['i'] = i_ins
def o_ovw(x, i):
    if len(i) != 2: return None
    n = i36(i[0])
    char = i[1]
    if n < 0:
        n += len(x)
    if n < 0 or n >= len(x):
        return None
    return x[:n] + char + x[n+1:]
FUNCTS['o'] = o_ovw
def apostrophe(x, i):
    n = i36(i)
    if n < 0:
        n += len(x) + 1
    if n < 0 or n > len(x):
        return None
    return x[:n]
FUNCTS["'"] = apostrophe
FUNCTS['s'] = lambda x, i: x.replace(i[0], i[1]) if len(i) == 2 else None
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
    if n < 0 or m < 0: return None
    if n < 0: n += len(x)
    if m < 0: m += len(x)
    if n >= len(x) or m >= len(x): return None
    chars = list(x)
    chars[n], chars[m] = chars[m], chars[n]
    return ''.join(chars)
FUNCTS['*'] = star
def L(x, i):
    n = i36(i)
    if n < 0: n += len(x)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    shifted_val = char_val << 1
    return x[:n] + chr(shifted_val) + x[n+1:]
FUNCTS['L'] = L
def R(x, i):
    n = i36(i)
    if n < 0: n += len(x)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    shifted_val = char_val >> 1
    return x[:n] + chr(shifted_val) + x[n+1:]
FUNCTS['R'] = R
def plus(x, i):
    n = i36(i)
    if n < 0: n += len(x)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    return x[:n] + chr(char_val + 1) + x[n+1:]
FUNCTS['+'] = plus
def minus(x, i):
    n = i36(i)
    if n < 0: n += len(x)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    return x[:n] + chr(char_val - 1) + x[n+1:]
FUNCTS['-'] = minus
def dot(x, i):
    n = i36(i)
    if n < 0: n += len(x)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    return x[:n] + chr(char_val + 1) + x[n+1:]
FUNCTS['.'] = dot
def comma(x, i):
    n = i36(i)
    if n < 0: n += len(x)
    if n < 0 or n >= len(x): return None
    char_val = ord(x[n])
    return x[:n] + chr(char_val - 1) + x[n+1:]
FUNCTS[','] = comma
def y(x, i):
    n = i36(i)
    if n < 0: n += len(x) + 1
    if n < 0 or n > len(x):
        return None
    return x[:n] + x
FUNCTS['y'] = y
def Y(x, i):
    n = i36(i)
    if n < 0: n += len(x) + 1
    if n < 0 or n > len(x):
        return None
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
    if len(i) < 2: return None
    n = i36(i[0])
    separator = i[1]
    parts = x.split(separator)
    if n < 0: n += len(parts)
    if n < 0 or n >= len(parts): return None
    parts[n] = parts[n].swapcase()
    return separator.join(parts)
FUNCTS['3'] = three

cdef class RuleEngine:
    cpdef apply_rule(self, str rule_str, str word):
        cdef list parsed_rule = self._parse_rule(rule_str)
        if parsed_rule is None:
            return None
        
        cdef str current_word = word
        cdef tuple function
        for function in parsed_rule:
            if current_word is None:
                return None
            try:
                current_word = FUNCTS[function[0]](current_word, function[1])
            except (IndexError, ValueError, TypeError):
                return None
        return current_word

    cdef list _parse_rule(self, str rule_str):
        cdef list rules = []
        cdef int i = 0
        cdef int length = len(rule_str)
        cdef str char
        cdef str arg1
        cdef str arg2
        while i < length:
            char = rule_str[i]
            if char in "ludtrf{}qkK":
                rules.append((char, ""))
                i += 1
            elif char in "$^":
                if i + 1 < length:
                    rules.append((char, rule_str[i+1]))
                    i += 2
                else:
                    return None
            elif char in "CcE":
                rules.append((char, ""))
                i += 1
            elif char in "[]":
                rules.append((char, ""))
                i += 1
            elif char in "opsx*+-.yY":
                if i + 2 <= length:
                    rules.append((char, rule_str[i+1:i+3]))
                    i += 3
                else:
                    return None
            elif char in "D'LRTZz":
                if i + 1 < length:
                    rules.append((char, rule_str[i+1]))
                    i += 2
                else:
                    return None
            elif char in "@":
                if i + 1 < length:
                    rules.append((char, rule_str[i+1]) if i + 1 < length else None)
                    i += 2
                else:
                    return None
            elif char == 'e':
                if i + 2 <= length:
                    rules.append((char, rule_str[i+1]))
                    i += 2
                else:
                    return None
            elif char == '3':
                if i + 2 <= length:
                    rules.append((char, rule_str[i+1:i+3]))
                    i += 3
                else:
                    return None
            else:
                return None
        return rules

def simplify_rules(rules):
    simplified = Counter()
    for rule, count in rules.items():
        # Check for T0T0 redundant rules
        if 'T0T0' in rule:
            rule = rule.replace('T0T0', '')
        
        # Check for '' redundant rules
        if rule == "''":
            rule = ""
        
        # If the rule becomes empty after simplification, replace it with ':'
        if not rule:
            rule = ":"
        
        simplified[rule] += count
    return simplified

def process_word_optimized(args):
    """
    Optimization of rule extraction using caching.
    Builds rule chains iteratively, avoiding redundant calculations.
    """
    word, rules, target_set, rule_engine, chain_depth = args
    found_rules_local = Counter()
    
    # Dictionary to store intermediate words and the rule chains that lead to them
    # Format: { resulting_word: [list_of_rule_chains] }
    intermediate_results = {word: [""]} # Initial state: base word with an empty chain
    
    for depth in range(1, chain_depth + 1):
        # New results that will serve as the base for the next depth
        new_intermediate_results = {}
        for prev_word, rule_chains in intermediate_results.items():
            for rule in rules:
                transformed_word = rule_engine.apply_rule(rule, prev_word)
                if transformed_word is not None and transformed_word not in new_intermediate_results:
                    # Create a new, full rule chain
                    for prev_chain in rule_chains:
                        new_chain = prev_chain + rule
                        
                        # Check if the newly transformed word is in the target set
                        if transformed_word != word and transformed_word in target_set:
                            found_rules_local[new_chain] += 1
                        
                        # Add the result to the cache for the next level
                        if transformed_word not in new_intermediate_results:
                            new_intermediate_results[transformed_word] = []
                        new_intermediate_results[transformed_word].append(new_chain)
        
        intermediate_results = new_intermediate_results

    return found_rules_local

cpdef load_data(str filename):
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist.")
        return None
    try:
        with open(filename, 'r', encoding='latin-1') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"An error occurred while loading the file '{filename}': {e}")
        return None

cpdef encode_non_ascii_to_hex(str rule):
    cdef str hex_encoded_rule = ""
    cdef str char
    try:
        rule.encode('ascii')
        return rule
    except UnicodeEncodeError:
        for char in rule:
            if char in "ludtrf{}q[]CcE$":
                hex_encoded_rule += char
            else:
                hex_encoded_rule += "".join([f"\\x{ord(byte):02x}" for byte in char.encode('latin-1')])
        return hex_encoded_rule

def main():
    parser = argparse.ArgumentParser(description='Hashcat rule extractor based on example passwords.')
    parser.add_argument('-b', '--base-words', help='Path to the file with base words (required unless --autobase is used).')
    parser.add_argument('-t', '--target-passwords', required=True, help='Path to the file with target passwords.')
    parser.add_argument('-o', '--output-file', default='extracted_rules.rule', help='Path to the output file for the rules. Default: extracted_rules.rule')
    parser.add_argument('--chain-depth', type=int, default=1, help='Number of rules to chain together. Use with caution. Default: 1 (single rules)')
    parser.add_argument('--autobase', action='store_true', help='Automatically generate base words from the target list (memory-intensive).')
    
    args = parser.parse_args()

    print("--- Hashcat Rule Extractor ---")
    
    if args.autobase:
        print("Mode: Autobase (generating base words from the target list)")
        print("WARNING: The --autobase option loads the entire target file into RAM, which can consume a large amount of memory on larger files.")
    
    if args.chain_depth > 1:
        print(f"Mode: Chain Extraction (Depth: {args.chain_depth})")
        print("WARNING: A caching optimization has been applied. This mode consumes significantly more RAM but runs much faster than the standard version.")
    
    print("-----------------------------------------------------")

    if not args.autobase and not args.base_words:
        parser.error('argument -b/--base-words is required unless --autobase is specified.')

    # Load target passwords first, as they are needed in both modes
    target_passwords = load_data(args.target_passwords)
    if target_passwords is None:
        return
    
    target_set = set(target_passwords)
    
    if args.autobase:
        # Generate base_words from target_passwords
        print("Generating base words from the target list...")
        base_words = [word for word in target_passwords if word.isalpha()]
    else:
        # Load base_words from the specified file
        base_words = load_data(args.base_words)
    
    if base_words is None:
        return
        
    print(f"Loaded {len(base_words)} base words.")
    print(f"Loaded {len(target_passwords)} target passwords.")
    
    simple_rules = set()
    simple_rules.add(':')
    simple_rules.update(['l', 'u', 'c', 'C', 't', 'r', 'f', 'd', 'q', '{', '}', '[', ']', 'k', 'K', 'E'])
    
    for i in range(2, 5): simple_rules.add('p' + str(i))
    for i in range(1, 4): simple_rules.add(']' * i)
    for i in range(1, 4): simple_rules.add('[' * i)
    
    rule_engine = RuleEngine()
    
    chars_to_test = string.digits + string.ascii_letters + string.punctuation
    for char in chars_to_test:
        simple_rules.add(f"${char}")
        simple_rules.add(f"^{char}")
        
    positions = [str(i) for i in range(16)]
    for pos in positions:
        simple_rules.add(f"T{pos}")
        simple_rules.add(f"D{pos}")
        simple_rules.add(f"'{pos}")
        simple_rules.add(f"L{pos}")
        simple_rules.add(f"R{pos}")
        simple_rules.add(f"+{pos}")
        simple_rules.add(f"-{pos}")
        simple_rules.add(f".{pos}")
        simple_rules.add(f",{pos}")
        simple_rules.add(f"y{pos}")
        simple_rules.add(f"Y{pos}")
        simple_rules.add(f"z{pos}")
        simple_rules.add(f"Z{pos}")
    
    replacements = {'a':'@', 'e':'3', 'i':'1', 'o':'0', 's':'$'}
    for k, v in replacements.items():
        simple_rules.add(f"s{k}{v}")
    
    for pos_a in positions:
        for pos_b in positions:
            if pos_a != pos_b:
                simple_rules.add(f"*{pos_a}{pos_b}")

    print(f"Generated a set of {len(simple_rules)} rules for testing.")
    print("Starting rule extraction...")

    task_args = [(word, list(simple_rules), target_set, rule_engine, args.chain_depth) for word in base_words]
    
    found_rules = Counter()
    
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel processing.")
    
    # We use the new, optimized function
    with Pool(processes=num_processes) as p:
        results = p.imap_unordered(process_word_optimized, task_args, chunksize=100)
        
        for result in tqdm(results, total=len(base_words), desc="Processing words"):
            found_rules.update(result)

    print("-----------------------------")
    if not found_rules:
        print("No matching rules found.")
        return
    
    print(f"Found {len(found_rules)} unique rules.")

    simplified_rules = simplify_rules(found_rules)
    print(f"Simplified rules down to {len(simplified_rules)} unique rules.")

    print("\n--- Top 10 Most Frequent Rules ---")
    for rule, count in simplified_rules.most_common(10):
        print(f"  {rule:<15} (Occurrences: {count})")
    
    print("\n-----------------------------")

    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for rule, count in simplified_rules.most_common():
                hex_rule = encode_non_ascii_to_hex(rule)
                f.write(f"{hex_rule}\n")
        print(f"All simplified rules, sorted by frequency, were successfully saved to the file '{args.output_file}'.")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
        
if __name__ == "__main__":
    main()
