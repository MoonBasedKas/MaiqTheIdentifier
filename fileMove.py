#!/usr/bin/env python3

import sys
import os
import shutil

def main():
    moveFiles(".\\faces", ".\\data")

def moveFiles(target, dest):
    dirs = os.listdir(target)
    temp = []
    loc = ""
    for folder in dirs:
        loc = target+os.sep+folder
        temp = os.listdir(loc)
        for file in temp:
            shutil.copy(loc + os.sep + file, dest + os.sep + file)




if __name__ == "__main__":
    main()