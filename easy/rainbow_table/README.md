# FNV1a Hash

## Description
Implement a program that applies the FNV1a hash function to an array of 32-bit integers on a GPU. The program should take an input array and a number of iterations R, then produce an output array where each element is hashed R times using the FNV1a algorithm.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in the output array

## Input Description
You will be given 2 values N and R, followed by N integer values.

Input format:
```bash
N R
a1 a2 ... aN
```

Constraints:
- 1 ≤ N ≤ 10000000, Length of array(integer)
- 1 ≤ R ≤ 1000, Number of iterations(integer)
- -2^31 ≤ ai ≤ 2^31-1, Array values(integer)

## Output Description
Output N unsigned integer values representing the FNV1a hash results after R iterations, separated by spaces, with a newline at the end.

Output format:
```bash
hash(a1) hash(a2) ... hash(aN)
```

Where each hash(ai) is the result of applying FNV1a hash function R times to ai.

## Example

### Input
```
3 2
123 456 789
```

### Output
```
3728671011 2847294259 1965917507
```