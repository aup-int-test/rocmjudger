# Color Inversion

## Description
Implement a program that performs image color inversion on a GPU. The program should take an input image represented as RGBA pixels and produce an output image where each RGB channel is inverted (255 - original_value). The alpha channel remains unchanged.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in the image array

## Input Description
You will be given 2 values: width and height, followed by width×height×4 values representing RGBA pixel data.

Input format:
```bash
width height
r1 g1 b1 a1 r2 g2 b2 a2 ... r_n g_n b_n a_n
```

Constraints:
- 1 ≤ width, height ≤ 10000, Image dimensions(integer)
- 0 ≤ ri, gi, bi, ai ≤ 255, Pixel channel values(unsigned char)

## Output Description
Output width×height×4 integer values representing the inverted RGBA pixel data, with 4 values per line (one pixel per line).

Output format:
```bash
inverted_r1 inverted_g1 inverted_b1 a1
inverted_r2 inverted_g2 inverted_b2 a2
...
inverted_r_n inverted_g_n inverted_b_n a_n
```

Where inverted_channel = 255 - original_channel for RGB channels, and alpha channel remains unchanged.

## Example

### Input
```
2 1
100 150 200 255 50 75 25 128
```

### Output
```
155 105 55 255
205 180 230 128
```