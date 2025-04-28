#!/usr/bin/env python3

import sys
import os

def main():
    argv = sys.argv
    label = 0
    dirs = []
    out = "labels.csv"
    dest = "." + os.sep
    while argv != []:
        temp = argv.pop(0)
        if temp == "-help":
            print("-label [number]: Creates a label for this dataset")
            print("-dir [name]: What directory are we taking from")
            print("-dest [name]: Where are the folders being moved to")
            print("-o [name]: what should the label file be named")
        elif temp == "-label":
            label = int(argv.pop(0))
        elif temp == "-dir":
            dirs.append(argv.pop(0))
        elif temp == "-dest":
            dest = argv.pop(0)
        elif temp == "-o":
            out = argv.pop(0)
    files = []
    while dirs != []:
        files += getDir(dirs.pop(0))

    if dest[-1] != os.sep:
        dest += os.sep + out
    else:
        dest += out
    writeLabels(files, label, dest)

    

def writeLabels(files, label, dest):
    fp = open(dest, "a")
    for i in files:
        fp.write(f"{i},{label}\n")
    fp.close()


def getDir(dirs):
    d = os.listdir(dirs)
    f = []
    for x in d:
        f.append(x)
    return f

if __name__ == "__main__":
    main()