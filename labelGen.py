#!/usr/bin/env python3

import sys
import os

def main():
    argv = sys.argv
    label = 0
    dirs = []
    out = "labels.csv"
    dest = "." + os.sep
    rec = False
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
        elif temp == "-rec":
            rec = True
    files = []
    if rec:
        recGrab(dirs, dest)
        return 0
    


    if dest[-1] != os.sep:
        dest += os.sep + out
    else:
        while dirs != []:
            files += getDir(dirs.pop(0))
        dest += out
    writeLabels(files, label, dest)

def recGrab(d, dest):
    label = 0
    print(d)
    dirs = os.listdir(d[0])
    files = []
    temp = []
    for folder in dirs:
        temp = os.listdir(d[0] + folder)
        for file in temp:
            files.append([file, label])
        label += 1
    fp = open(dest, "w")
    for i in files:
        fp.write(f"{i[0]},{i[1]}\n")
    fp.close()


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