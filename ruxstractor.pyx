# ruxstractor-v1.2.pyx

# distutils: extra_compile_args=-O3
# distutils: language=c
import os
import argparse
import sys
import string
import itertools
import numpy as np
import tempfile
import subprocess
import psutil
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial

# RAM monitoring settings (change if needed)
MAX_RAM_PERCENTAGE = 90.0

# The following Cython imports are used for C-level optimization.
from cython.cimports.libc.stdlib import strtol
from cython.cimports.libc.string import strlen

def check_ram_and_exit():
    """Checks RAM usage and terminates the process if it exceeds the threshold."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_used_bytes = mem_info.rss
    total_ram_bytes = psutil.virtual_memory().total
    ram_percentage = (ram_used_bytes / total_ram_bytes) * 100
    
    if ram_percentage > MAX_RAM_PERCENTAGE:
        print(f"\n[!] WARNING: RAM usage ({ram_percentage:.2f}%) exceeds the set limit ({MAX_RAM_PERCENTAGE:.1f}%).")
        print("Terminating process to prevent system instability.")
        os._exit(1)

# This is a Python wrapper, the C-optimized version is below.
def i36(s):
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

# --- Levenshtein Algorithm and Rule Generation (New Cython-optimized section) ---

cdef int[:, :] levenshtein_matrix(str s1, str s2):
    """
    Calculates the Levenshtein distance matrix between two strings.
    This implementation is optimized for Cython.
    """
    cdef int len1 = len(s1)
    cdef int len2 = len(s2)
    cdef int i, j, insertions, deletions, substitutions

    cdef int[:, :] matrix = np.zeros((len1 + 1, len2 + 1), dtype=np.int32)

    for i in range(len1 + 1):
        matrix[i, 0] = i
    for j in range(1, len2 + 1):
        matrix[0, j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            insertions = matrix[i, j - 1] + 1
            deletions = matrix[i - 1, j] + 1
            substitutions = matrix[i - 1, j - 1] + (s1[i - 1] != s2[j - 1])
            matrix[i, j] = min(insertions, deletions, substitutions)
    
    return matrix

cdef list levenshtein_reverse_recursive(int[:, :] matrix, int i, int j, int path_len, int max_dist):
    """
    Recursively finds all shortest paths from the end of the matrix to the start.
    This is a recursive function, so Python objects are used for the path list.
    """
    # Cython fix: All cdef declarations must be at the top of the function
    cdef int cost = matrix[i, j]
    cdef long cost_delete = sys.maxsize
    cdef long cost_insert = sys.maxsize
    cdef long cost_equal_or_replace = sys.maxsize
    cdef long cost_min

    if (i == 0 and j == 0) or path_len > max_dist:
        return [[]]
    else:
        paths = []
        
        if i > 0:
            cost_insert = matrix[i - 1, j]
        if j > 0:
            cost_delete = matrix[i, j - 1]
        if i > 0 and j > 0:
            cost_equal_or_replace = matrix[i - 1, j - 1]
        
        cost_min = min(cost_delete, cost_insert, cost_equal_or_replace)
        
        if cost_insert == cost_min:
            insert_paths = levenshtein_reverse_recursive(matrix, i - 1, j, path_len + 1, max_dist)
            for p in insert_paths:
                paths.append(p + [('insert', i, j)])

        if cost_delete == cost_min:
            delete_paths = levenshtein_reverse_recursive(matrix, i, j - 1, path_len + 1, max_dist)
            for p in delete_paths:
                paths.append(p + [('delete', i, j)])

        if cost_equal_or_replace == cost_min:
            if cost_equal_or_replace == cost:
                equal_paths = levenshtein_reverse_recursive(matrix, i - 1, j - 1, path_len, max_dist)
                for p in equal_paths:
                    paths.append(p)
            else:
                replace_paths = levenshtein_reverse_recursive(matrix, i - 1, j - 1, path_len + 1, max_dist)
                for p in replace_paths:
                    paths.append(p + [('replace', i, j)])

        return paths

cpdef str int_to_hashcat(int n):
    """
    Converts an integer position to a hashcat character representation.
    """
    if 0 <= n < 10:
        return str(n)
    elif 10 <= n < 36:
        return chr(65 + n - 10)
    else:
        return ""

def generate_rules_from_path(str word, str password, list path):
    """
    Generates a hashcat-compatible rule string from a Levenshtein path.
    """
    rules = []
    
    # Simple, non-positional rules (l, u, c, C, t, r)
    if password == word.lower():
        rules.append('l')
        word = password
    elif password == word.upper():
        rules.append('u')
        word = password
    elif password == word.capitalize():
        rules.append('c')
        word = password
    elif password == word.swapcase():
        rules.append('t')
        word = password
    elif password == word[::-1]:
        rules.append('r')
        word = password
        
    for op, r, c in path:
        if op == 'insert':
            rules.append(f'i{int_to_hashcat(c-1)}{password[c-1]}')
        elif op == 'delete':
            rules.append(f'D{int_to_hashcat(r-1)}')
        elif op == 'replace':
            pos_w = r - 1
            pos_p = c - 1
            if word[pos_w].islower() and password[pos_p].isupper():
                rules.append(f'T{int_to_hashcat(pos_w)}')
            elif word[pos_w].isupper() and password[pos_p].islower():
                rules.append(f'T{int_to_hashcat(pos_w)}')
            elif word[pos_w].isalpha() and password[pos_p].isalpha() and word[pos_w].lower() == password[pos_p].lower():
                rules.append(f's{word[pos_w]}{password[pos_p]}')
            else:
                rules.append(f'o{int_to_hashcat(pos_w)}{password[pos_p]}')

    # Remove duplicates and simplify rule chains
    simplified_rules = []
    for rule in rules:
        if rule in ['l', 'u', 'c', 't', 'r']:
            if rule not in simplified_rules:
                simplified_rules.insert(0, rule)
        else:
            simplified_rules.append(rule)

    return "".join(simplified_rules)

# This worker function is now simpler, as it just calls the core Levenshtein functions
cpdef find_levenshtein_rules_worker(tuple args):
    """
    A worker function to find rules for a given pair of words using Levenshtein.
    """
    cdef str base = args[0]
    cdef str target = args[1]
    cdef int max_depth = args[2]

    if not base or not target or base == target:
        return None

    matrix = levenshtein_matrix(base, target)
    if matrix[len(base), len(target)] > max_depth:
        return None

    paths = levenshtein_reverse_recursive(matrix, len(base), len(target), 0, max_dist=max_depth)
    
    if not paths:
        return None

    paths.sort(key=len)
    shortest_path = paths[0]
    
    rule_string = generate_rules_from_path(base, target, shortest_path)

    return rule_string

# New worker function for append rules
cpdef find_append_rules_worker(tuple args):
    """
    A worker function to find append rules ($) for a given pair of words.
    """
    cdef str base = args[0]
    cdef str target = args[1]
    
    if target.startswith(base) and len(target) > len(base):
        appended_chars = target[len(base):]
        rule = "$" + "$".join(appended_chars)
        return rule
    
    return None

# New worker function for prepend rules
cpdef find_prepend_rules_worker(tuple args):
    """
    A worker function to find prepend rules (^) for a given pair of words.
    """
    cdef str base = args[0]
    cdef str target = args[1]
    
    if base.endswith(target) and len(base) > len(target):
        prepended_chars = base[:-len(target)]
        rule = "^" + "^".join(prepended_chars)
        return rule
    
    return None

# New worker for 'zN' rules
cpdef find_z_rules_worker(tuple args):
    cdef str base = args[0]
    cdef str target = args[1]
    if not base or not target or len(target) <= len(base):
        return None
    
    cdef str first_char = base[0]
    cdef int num_duplicated = len(target) - len(base)
    
    if target == first_char * num_duplicated + base:
        return f"z{int_to_hashcat(num_duplicated)}"
    
    return None

# New worker for 'ZN' rules
cpdef find_Z_rules_worker(tuple args):
    cdef str base = args[0]
    cdef str target = args[1]
    if not base or not target or len(target) <= len(base):
        return None
        
    cdef str last_char = base[-1]
    cdef int num_duplicated = len(target) - len(base)

    if target == base + last_char * num_duplicated:
        return f"Z{int_to_hashcat(num_duplicated)}"

    return None

# New worker for 'xNM' rules
cpdef find_x_rules_worker(tuple args):
    cdef str base = args[0]
    cdef str target = args[1]
    
    cdef int len_base = len(base)
    cdef int len_target = len(target)
    
    if len_target > len_base:
        return None
    
    cdef int i, j
    for i in range(len_base):
        for j in range(len_base - i + 1):
            if base[i:i+j] == target:
                return f"x{int_to_hashcat(i)}{int_to_hashcat(j)}"

    return None

# New worker for 'oNX' rules
cpdef find_o_rules_worker(tuple args):
    cdef str base = args[0]
    cdef str target = args[1]

    cdef int len_base = len(base)
    cdef int len_target = len(target)
    cdef int diff_count = 0
    cdef int diff_pos = -1

    if len_base != len_target or len_base == 0:
        return None

    cdef int i
    for i in range(len_base):
        if base[i] != target[i]:
            diff_count += 1
            diff_pos = i

    if diff_count == 1:
        return f"o{int_to_hashcat(diff_pos)}{target[diff_pos]}"

    return None

# --- Rest of the script (maintained from previous version) ---

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
def f(x, i):
    return x+x[::-1]
FUNCTS['f'] = f
def br_open(x, i):
    return x[1:]+x[0] if len(x) > 0 else x
FUNCTS['{'] = br_open
def br_close(x, i):
    return x[-1]+x[:-1] if len(x) > 0 else x
FUNCTS['}'] = br_close
FUNCTS['$'] = lambda x, i: x+i
FUNCTS['^'] = lambda x, i: i+x
def br_sq_open(x, i):
    return x[1:]
FUNCTS['['] = br_sq_open
def br_sq_close(x, i):
    return x[:-1]
FUNCTS[']'] = br_sq_close
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
    if n < 0: n += len(x) + 1
    if n < 0 or n > len(x):
        return None
    return x[:n]
FUNCTS["'"] = apostrophe
def s_sub(x, i):
    if len(i) != 2: return None
    return x.replace(i[0], i[1])
FUNCTS['s'] = s_sub
def at(x, i):
    if len(i) == 0: return None
    return x.replace(i, '')
FUNCTS['@'] = at
def z(x, i):
    number = i36(i)
    if number < 0: return None
    return x[0]*number + x if len(x) > 0 else x
FUNCTS['z'] = z
def Z(x, i):
    number = i36(i)
    if number < 0: return None
    return x+x[-1]*number if len(x) > 0 else x
FUNCTS['Z'] = Z
def q(x, i):
    return ''.join([c*2 for c in x])
FUNCTS['q'] = q
def k_swap(x, i):
    if len(x) < 2: return x
    return x[1]+x[0]+x[2:]
FUNCTS['k'] = k_swap
def K_swap(x, i):
    if len(x) < 2: return x
    return x[:-2]+x[-1]+x[-2]
FUNCTS['K'] = K_swap
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
def E_title(x, i):
    return string.capwords(x, ' ')
FUNCTS['E'] = E_title
def e_title_sep(x, i):
    if len(i) != 1: return None
    return string.capwords(x, i)
FUNCTS['e'] = e_title_sep
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
        if 'T0T0' in rule:
            rule = rule.replace('T0T0', '')
        
        if rule == "''":
            rule = ""
        
        if not rule:
            rule = ":"
        
        simplified[rule] += count
    return simplified

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
                hex_encoded_rule += "".join([f"\\x{byte:02x}" for byte in char.encode('latin-1')])
        return hex_encoded_rule

def generate_levenshtein_tasks_to_file(temp_tasks_path, base_words, target_file, start_depth, end_depth):
    total_tasks = 0
    with open(temp_tasks_path, 'w', encoding='utf-8') as f:
        with open(target_file, 'r', encoding='latin-1') as f_target:
            for depth in range(start_depth, end_depth + 1):
                f_target.seek(0) # Reset file pointer for each depth
                for target in f_target:
                    target = target.strip()
                    if target:
                        for base in base_words:
                            if base != target:
                                f.write(f"{base}\t{target}\t{depth}\n")
                                total_tasks += 1
    return total_tasks

def levenshtein_task_generator(temp_tasks_path):
    with open(temp_tasks_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                yield (parts[0], parts[1], int(parts[2]))

def count_rules_from_file(temp_rules_path):
    found_rules = Counter()
    with open(temp_rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            rule = line.strip()
            if rule:
                found_rules[rule] += 1
    return found_rules

def main():
    parser = argparse.ArgumentParser(description='Hashcat rule extractor based on example passwords.')
    parser.add_argument('-b', '--base-words', help='Path to the file with base words (required unless --autobase is used).')
    parser.add_argument('-t', '--target-passwords', required=True, help='Path to the file with target passwords.')
    parser.add_argument('-o', '--output-file', default='extracted_rules.rule', help='Path to the output file for the rules. Default: extracted_rules.rule')
    parser.add_argument('--depth-range', default='1-1', help='Range of Levenshtein distances to find rules for, e.g., "1-5". Default: "1-1"')
    parser.add_argument('--autobase', action='store_true', help='Automatically generate base words from the target list (memory-intensive).')
    parser.add_argument('--in-ram', action='store_true', help='Use RAM for all processing. Faster but may consume a lot of RAM.')
    parser.add_argument('--cleanup-tool', help='Path to cleanup-rules.bin to post-process the rules.')
    
    args = parser.parse_args()

    # Parse the depth range first, as it's used in print statements
    try:
        depth_parts = args.depth_range.split('-')
        if len(depth_parts) != 2:
            raise ValueError("Invalid format for --depth-range. Please use 'start-end', e.g., '1-5'.")
        start_depth = int(depth_parts[0])
        end_depth = int(depth_parts[1])
        if start_depth > end_depth or start_depth < 1:
            raise ValueError("Invalid range. Start depth must be at least 1 and less than or equal to end depth.")
    except (ValueError, IndexError) as e:
        print(f"Error parsing --depth-range: {e}")
        sys.exit(1)

    print("--- Hashcat Rule Extractor ---")
    check_ram_and_exit()

    # Check if a dedicated base words file is provided
    if not args.autobase and not args.base_words:
        parser.error('argument -b/--base-words is required unless --autobase is specified.')

    base_words = None
    target_passwords = None
    temp_base_path = None
    
    # Determine processing mode and load base words
    # The condition was changed so that the --in-ram flag always forces in-memory mode,
    # regardless of file size
    if args.in_ram:
        print("Using memory-intensive mode (--in-ram). This is faster but may consume a lot of RAM.")
        check_ram_and_exit()
        target_passwords = load_data(args.target_passwords)
        if target_passwords is None: return
        print(f"Loaded {len(target_passwords)} target passwords.")
        if args.autobase:
            base_words = [word for word in target_passwords if word.isalpha()]
            print("Mode: Autobase (generating base words from the target list)")
            print(f"Loaded {len(base_words)} base words.")
        else:
            base_words = load_data(args.base_words)
            if base_words is None: return
            print(f"Loaded {len(base_words)} base words.")
            
    else: # File-based mode (default)
        print("Using file-based mode (default). This is slower but very low on RAM usage.")
        
        # In file-based mode, the base list is loaded into memory, but the target list is streamed
        if args.autobase:
            print("Mode: Autobase (generating base words from the target list)")
            # In file-based mode, we need to create a temporary file for base words
            if not os.path.exists(args.target_passwords):
                print(f"Error: The target file '{args.target_passwords}' does not exist.")
                return
            with open(args.target_passwords, 'r', encoding='latin-1') as f_target, \
                    tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as temp_base_file_obj:
                temp_base_path = temp_base_file_obj.name
                for word in f_target:
                    word = word.strip()
                    if word and word.isalpha():
                        temp_base_file_obj.write(word + '\n')
            base_words = load_data(temp_base_path)
            if base_words is None: return
            print(f"Loaded {len(base_words)} auto-generated base words to a temporary file.")
        else:
            base_words = load_data(args.base_words)
            if base_words is None: return
            print(f"Loaded {len(base_words)} base words.")
        
        print(f"Target passwords file will be processed in a memory-optimized way.")
    
    print(f"Mode: Levenshtein Rule Extraction (Depth Range: {start_depth}-{end_depth})")
    print("-----------------------------------------------------")
    
    if not base_words:
        print("Error: No base words found. Cannot proceed.")
        return
        
    found_rules = Counter()
    total_tasks = 0
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel processing.")
    
    if args.in_ram:
        # --- Memory-intensive mode ---
        check_ram_and_exit()
        
        # Non-Levenshtein rules
        word_pairs = []
        if args.autobase:
            word_pairs.extend(list(itertools.permutations(set(base_words), 2)))
        else:
            word_pairs.extend(list(itertools.product(set(base_words), set(target_passwords))))
            
        print("Finding non-Levenshtein special rules...")
        with Pool(processes=num_processes) as p:
            check_ram_and_exit()
            results = p.imap_unordered(find_append_rules_worker, word_pairs, chunksize=100)
            for rule in tqdm(results, total=len(word_pairs), desc="Finding append ($) rules"):
                if rule: found_rules[rule] += 1
            check_ram_and_exit()
            results = p.imap_unordered(find_prepend_rules_worker, word_pairs, chunksize=100)
            for rule in tqdm(results, total=len(word_pairs), desc="Finding prepend (^) rules"):
                if rule: found_rules[rule] += 1
            check_ram_and_exit()
            results = p.imap_unordered(find_z_rules_worker, word_pairs, chunksize=100)
            for rule in tqdm(results, total=len(word_pairs), desc="Finding duplicate first char (z) rules"):
                if rule: found_rules[rule] += 1
            check_ram_and_exit()
            results = p.imap_unordered(find_Z_rules_worker, word_pairs, chunksize=100)
            for rule in tqdm(results, total=len(word_pairs), desc="Finding duplicate last char (Z) rules"):
                if rule: found_rules[rule] += 1
            check_ram_and_exit()
            results = p.imap_unordered(find_x_rules_worker, word_pairs, chunksize=100)
            for rule in tqdm(results, total=len(word_pairs), desc="Finding extract range (x) rules"):
                if rule: found_rules[rule] += 1
            check_ram_and_exit()
            results = p.imap_unordered(find_o_rules_worker, word_pairs, chunksize=100)
            for rule in tqdm(results, total=len(word_pairs), desc="Finding overwrite (o) rules"):
                if rule: found_rules[rule] += 1
        
        # Levenshtein rules
        print("\nGenerating Levenshtein tasks in RAM...")
        levenshtein_tasks = []
        for depth in range(start_depth, end_depth + 1):
            if args.autobase:
                levenshtein_tasks.extend([(base, target, depth)
                                          for base, target in itertools.permutations(base_words, 2)
                                          if base != target])
            else:
                levenshtein_tasks.extend([(base, target, depth)
                                          for base, target in itertools.product(base_words, target_passwords)
                                          if base != target])
        total_tasks = len(levenshtein_tasks)
        print(f"Total Levenshtein tasks to process: {total_tasks}")
        
        check_ram_and_exit()
        with Pool(processes=num_processes) as p:
            results = p.imap_unordered(find_levenshtein_rules_worker, levenshtein_tasks, chunksize=100)
            for result in tqdm(results, total=total_tasks, desc="Processing Levenshtein"):
                if result:
                    found_rules[result] += 1

    else:
        # --- File-based, memory-optimized mode ---
        
        temp_tasks_path = 'rux_levenshtein_tasks.tmp'
        temp_rules_path = 'rux_levenshtein_rules.tmp'
        
        try:
            print("\nGenerating Levenshtein tasks and saving to a temporary file...")
            total_tasks = generate_levenshtein_tasks_to_file(temp_tasks_path, base_words, args.target_passwords, start_depth, end_depth)
            print(f"Total Levenshtein tasks to process: {total_tasks}")

            with Pool(processes=num_processes) as p, open(temp_rules_path, 'w', encoding='utf-8') as f_rules:
                task_gen = levenshtein_task_generator(temp_tasks_path)
                results = p.imap_unordered(find_levenshtein_rules_worker, task_gen, chunksize=100)
                for result in tqdm(results, total=total_tasks, desc="Processing Levenshtein"):
                    if result:
                        f_rules.write(result + '\n')
                
            print("Processing complete. Counting and simplifying Levenshtein rules...")
            levenshtein_rules = count_rules_from_file(temp_rules_path)
            found_rules.update(levenshtein_rules)

        finally:
            if os.path.exists(temp_tasks_path): os.remove(temp_tasks_path)
            if os.path.exists(temp_rules_path): os.remove(temp_rules_path)
            print("Temporary files cleaned up.")

    print("\n-----------------------------")
    if not found_rules:
        print("No matching rules found.")
        if temp_base_path and os.path.exists(temp_base_path): os.remove(temp_base_path)
        return
    
    print(f"Found {len(found_rules)} unique rules.")

    simplified_rules = simplify_rules(found_rules)
    print(f"Simplified rules down to {len(simplified_rules)} unique rules.")

    print("\n--- Top 10 Most Frequent Rules ---")
    for rule, count in simplified_rules.most_common(10):
        print(f"  {rule:<15} (Occurrences: {count})")
    
    print("\n-----------------------------")

    if args.cleanup_tool:
        try:
            with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as temp_file:
                for rule, count in simplified_rules.most_common():
                    hex_rule = encode_non_ascii_to_hex(rule)
                    temp_file.write(f"{hex_rule}\n")
            
            print(f"Running cleanup tool and saving to '{args.output_file}'...")
            with open(temp_file.name, 'r', encoding='utf-8') as infile, open(args.output_file, 'w', encoding='utf-8') as outfile:
                subprocess.run([args.cleanup_tool, '2'], stdin=infile, stdout=outfile, check=True)
            print(f"Successfully cleaned rules and saved to '{args.output_file}'.")
            
        except FileNotFoundError:
            print(f"Error: The cleanup tool '{args.cleanup_tool}' was not found.")
        except subprocess.CalledProcessError as e:
            print(f"Error: The cleanup tool '{args.cleanup_tool}' failed with exit code {e.returncode}.")
        except Exception as e:
            print(f"An unexpected error occurred during cleanup: {e}")
        finally:
            if 'temp_file' in locals() and os.path.exists(temp_file.name): os.remove(temp_file.name)
    else:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for rule, count in simplified_rules.most_common():
                    hex_rule = encode_non_ascii_to_hex(rule)
                    f.write(f"{hex_rule}\n")
            print(f"All simplified rules were successfully saved to the file '{args.output_file}'.")
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")
        
    if temp_base_path and os.path.exists(temp_base_path):
        os.remove(temp_base_path)
        print(f"Temporary base file '{temp_base_path}' cleaned up.")

if __name__ == "__main__":
    main()
