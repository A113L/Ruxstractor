<h2>Welcome to Ruxstractor v1.2:</h2>

**[Preview](https://dai.ly/k6soryDAyMQbHZDQEgw)**

1. Performance with Cython and Multiprocessing: Key functions, such as Levenshtein distance calculation, are optimized for speed. The script utilizes all available CPU cores, which significantly reduces the time required to generate rules for larger datasets.

2. Memory Optimization: The script automatically switches to a file-based mode for larger password lists (>1.5MB)(tested on 16GB RAM - 14GB of RAM and 17GB of swap space used after loading 2.2MB file in --autobase mode and bypass limits), which eliminates the risk of crashing due to insufficient RAM and allows for processing of larger input files.

3. Versatility of Generated Rules: The tool is not limited to Levenshtein (iDs). It recognizes and generates a wide range of Hashcat rules, including:

4. Basic transformations: case changes (l, u, c, t), reversal (r).

5. Operations on word ends: appending ($), prepending (^), and duplication of the first/last character (z, Z).

6. Complex operations: overwriting (o), deleting fragments (x), simple substitutions (s), and others.

7. User-Friendliness: It offers a simple command-line interface, and the built-in RuleEngine class allows for effective testing of the generated rules.

*Summary:*

Ruxstractor v1.2 is a solid and efficient tool that capably transforms datasets into useful rules for hash cracking. Thanks to its optimizations and wide range of generated rules, it is one of the most powerful tools of its kind.

```
python3 run.py -h
usage: run.py [-h] [-b BASE_WORDS] -t TARGET_PASSWORDS [-o OUTPUT_FILE] [--depth-range DEPTH_RANGE] [--autobase] [--in-memory] [--cleanup-tool CLEANUP_TOOL]

Hashcat rule extractor based on example passwords.

options:
  -h, --help            show this help message and exit
  -b BASE_WORDS, --base-words BASE_WORDS
                        Path to the file with base words (required unless --autobase is used).
  -t TARGET_PASSWORDS, --target-passwords TARGET_PASSWORDS
                        Path to the file with target passwords.
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Path to the output file for the rules. Default: extracted_rules.rule
  --depth-range DEPTH_RANGE
                        Range of Levenshtein distances to find rules for, e.g., "1-5". Default: "1-1"
  --autobase            Automatically generate base words from the target list (memory-intensive).
  --in-ram           Use RAM for all processing. Faster for small datasets but memory-intensive.
  --cleanup-tool CLEANUP_TOOL
                        Path to cleanup-rules.bin to post-process the rules.
```
                        
                        
*Note about --in-ram flag:*

**The --in-ram mode:** A high, one-time cost at the beginning (long task list generation), but the process is very fast afterward because all data is in memory. Can be used to bypass filesize limits.

**The default mode (without --in-ram):** There is no long, initial task generation phase. The process is streamed, which is much slower, but it uses minimal memory and is ideal for larger lists.



*Technical Program Description - Hashcat Rule Extractor*

This program is a specialized offensive security tool designed for the automated extraction of effective transformation rules for the Hashcat password cracking software. Its primary goal is to optimize dictionary attacks by creating custom rule sets that reflect the common password modification patterns found in specific datasets.

*Core Principle of Operation*

The program operates on a comparison-based principle. It analyzes how words from one dataset (base words) have been transformed to become passwords in a second dataset (target passwords). For example, if "password" is a base word and "P@ssw0rd1" is a target password, the program identifies the sequence of rules that led to this transformation.

*Key Components and Functionality*

Rule Engine: The core of the program is its built-in engine, which replicates a wide range of transformations available in Hashcat. This includes simple operations like:

- Case changes (l, u).
- Character additions at the beginning or end (^, $).
- Character substitutions (e.g., s for s/a/@, s/o/0).
- Rotational shifts ({, }).
- Rule composition.

*Rule Extraction: The program performs two main operations*

- Single Rule Extraction: It tests every single rule from its built-in database against each base word. If a transformed word matches a password in the target list, the corresponding rule is logged.
- Rule Chain Extraction (--depth-range): In chain mode, the program tests combinations of two or more rules (e.g., l + s/a/@). This is little slower but allows it to discover more complex and realistic password patterns that a single rule would miss, such as a capitalization and a substitution "P"assword" -> "p"assword" -> "p@ssword".
- Process Pool (Multiprocessing): The implementation of Python's multiprocessing module allows the program to distribute the task of processing base words across all available CPU cores. This is critical for performance, especially in --chains mode, where the number of operations is orders of magnitude higher.

*Output and Compatibility*

- Discovered rules are collected and counted, then sorted by their frequency (how many passwords they successfully cracked). The most frequent rules are considered the most valuable for future attacks
- The results are saved to a standard .rule file, in a format fully compatible with Hashcat, ensuring its direct usefulness in attacks.

*Tactical Application in Password Attacks*

This tool is invaluable for security analysts and penetration testers. It allows them to:

- Use data from public password breaches (e.g., company databases) to identify the unique password patterns of users.
- Generate a precise set of rules that are specifically tailored to a given target.
- Significantly increase the success rate of future dictionary attacks, particularly in scenarios where standard, generic rules are not effective enough.

<h3>Credits:</h3>

- https://github.com/mkb2091/PyRuleEngine/blob/master/PyRuleEngine.py - for rule engine library

- https://github.com/Hydraze/pack/blob/master/rulegen.py - for Levenshtein idea of extracting rules

- https://gemini.google.com/ - for putting the script together
