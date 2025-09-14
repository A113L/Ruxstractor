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
- Rule Chain Extraction (--chains): In chain mode, the program tests combinations of two rules (e.g., l + s/a/@). This is significantly slower but allows it to discover more complex and realistic password patterns that a single rule would miss, such as a capitalization and a substitution "P"assword" -> "p"assword" -> "p@ssword".
- Process Pool (Multiprocessing): The implementation of Python's multiprocessing module allows the program to distribute the task of processing base words across all available CPU cores. This is critical for performance, especially in --chains mode, where the number of operations is orders of magnitude higher.

*Output and Compatibility*

- Discovered rules are collected and counted, then sorted by their frequency (how many passwords they successfully cracked). The most frequent rules are considered the most valuable for future attacks
- The results are saved to a standard .rule file, in a format fully compatible with Hashcat, ensuring its direct usefulness in attacks.

*Tactical Application in Password Attacks*

This tool is invaluable for security analysts and penetration testers. It allows them to:

- Use data from public password breaches (e.g., company databases) to identify the unique password patterns of users.
- Generate a precise set of rules that are specifically tailored to a given target.
- Significantly increase the success rate of future dictionary attacks, particularly in scenarios where standard, generic rules are not effective enough.
