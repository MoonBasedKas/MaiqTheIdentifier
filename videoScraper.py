#!/usr/bin/env python3
import sys
import cv2
import time
import os

def main():
    temp = None
    argv = sys.argv
    dest = "." + os.sep
    name = "frame"
    files = []
    argv.pop(0)
    while(argv != []):
        temp = argv.pop(0)
        if temp == "-help":
            print("[name]: What video to break down")
            print("-dest [name]: Where to write each file")
            print("-name [name]: What each file will be named")
            print("-dir [name]: What dir to place into")
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
    sucs = 0
    t = time.time()
    t = int(t)
    face = False
    while (True):
        temp = dest
        success, frame = capture.read()
        
        if success:
            sucs += 1
            face = detectFace(frame)
            if face:
                if sucs >= 30:
                    temp += name + str(t) + "-" + str(frames) + ".jpg"
                    cv2.imwrite(temp, frame)
                    frames += 1
                    sucs = 0
    
        else:
            break
    
        
    
    capture.release()

"""
Detemines if there is a face in the video.
"""
def detectFace(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    try: 
        if type(face) == tuple:
            return False
    except:
        pass
    return True

if __name__ == "__main__":
    main()