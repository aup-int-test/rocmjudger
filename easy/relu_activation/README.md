# ReLU Activation

## Description
Implement a program that applies the **ReLU (Rectified Linear Unit)** activation function to an array of 32-bit floating point numbers on a GPU.  
The program should take an input array and produce an output array where each element is the result of applying the ReLU function.

### Algorithm
The ReLU function is defined as:

$$
\mathrm{ReLU}(x) = \max(0, x)
$$

For each element \(x\) in the input array:
- If \(x > 0\), return \(x\)
- Otherwise, return \(0\)

**Requirements**
- External libraries are **not permitted**
- The `solve` function signature must remain unchanged
- The final result must be stored in the output array

## Input
You will be given one integer `N`, followed by `N` floating point values.

**Format**
```

N
a1 a2 ... aN

```

Constraints:
- 1 ≤ N ≤ 10,000,000 (integer)
- aᵢ: Array values (float)

## Output
Output `N` floating point numbers representing the ReLU activation result, separated by spaces, with a newline at the end.

**Format**
```

relu(a1) relu(a2) ... relu(aN)

```

Where  
\[
\text{relu}(x) = \max(0, x)
\]

## Example
### Input
```

6
-2.5 1.3 0.0 -5.7 3.8 -1.0

```

### Output
```

0 1.3 0 0 3.8 0

```