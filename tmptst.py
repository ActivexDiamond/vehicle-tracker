import cv2
# read the image
video = cv2.VideoCapture("./test/videos1/traffic4.mp4")
#img = cv2.imread("./test/images1/ar1.jpg")

while True:
	nxt, frame = video.read()
	if not nxt: break
	cv2.imshow("frame", frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# show the image
#cv2.imshow(winname="Face", mat=img)
# Wait for a key press to exit
#cv2.waitKey(delay=0)
# Close all windows
#cv2.destroyAllWindows() 
