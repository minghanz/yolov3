import os
import argparse
import sys

def preprocess_enter(lines):
    ### the items of returned list of readlines has '\n' at the end except the last line
    for i, line in enumerate(lines):
        if line[-1]=='\n':
            lines[i] = line[:-1]
    return lines

def cvt2coco(custom_name):
    path_coco = 'data/coco.names'
    with open(path_coco) as f:
        lines = f.readlines()
    lines = preprocess_enter(lines)

    with open(custom_name) as f:
        lines_custom = f.readlines()
    lines_custom = preprocess_enter(lines_custom)

    custom_index_in_coco = [lines.index(custom) for custom in lines_custom]
    custom_index = list(range(len(lines_custom)))
    output = {}
    for custom_idx, coco_idx in zip(custom_index, custom_index_in_coco):
        output[coco_idx] = custom_idx

    print(output)
    # print(custom_index_in_coco)
    return output

if __name__ == "__main__":
    custom_name = sys.argv[1]
    cvt2coco(custom_name)