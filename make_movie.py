
import cv2
import os

video_name = 'water.avi'

frame = cv2.imread('frames/0.png')
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 8, (width,height))

#N = len(os.listdir('frames/'))
nums = range(540 + 750//2) + range(1290, 2060)
for i in nums:
    video.write(cv2.imread('frames/%d.png'%i))

video.release()


    
