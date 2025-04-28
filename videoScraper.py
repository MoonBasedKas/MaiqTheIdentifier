#!/usr/bin/env python3
import sys
import cv2
import os

def main():
    temp = None
    argv = sys.argv
    dest = "." + os.sep
    name = "frame"
    files = []
    while(argv != []):
        temp = argv.pop(0)
        if temp == "-help":
            print("[name]: What video to break down")
            print("-dest [name]: Where to write each file")
            print("-name [name]: What each file will be named")
        elif temp == "-dest":
            dest = argv.pop(0)
        elif temp == "-dir":
            files += getDir(argv.pop(0))
        elif temp == "-name":
            name = argv.pop(0)
        else:
            files.append(temp)

    while(files != []):
        print(files)
        ripAndTear(files.pop(0), dest, name)


"""
Does it for a specific video
"""
def getDir(dirs):
    d = os.listdir(dirs)
    f = []
    for x in d:
        f.append(dirs + os.sep + x)
    return f


"""
Gets the frames of the video
src - video
dest - directory to write to
name - what to name each file
"""
def ripAndTear(src, dest, name):
    capture = cv2.VideoCapture(src)
    temp = ""
    frames = 1
    
    while (True):
        temp = dest
        success, frame = capture.read()
    
        if success:
            temp += name + str(frames) + ".jpg"
            cv2.imwrite(temp, frame)
            frames += frames
    
        else:
            break
    
        
    
    capture.release()

if __name__ == "__main__":
    main()