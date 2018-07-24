
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import turtle
import os
import math
import random
import pyaudio

def eye_aspect_ratio(eye):
	
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	
	C = dist.euclidean(eye[0], eye[3])

	
	ear = (A + B) / (2.0 * C)

	
	return ear
 

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3


COUNTER = 0
TOTAL = 0


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)


wn = turtle.Screen()
wn.bgcolor("black")
wn.title("Space Invaders")


border_pen = turtle.Turtle()
border_pen.speed(0)
border_pen.color("white")
border_pen.penup()
border_pen.setposition(-300,-300)
border_pen.pendown()
border_pen.pensize(3)
for side in range(4):
	border_pen.fd(600)
	border_pen.lt(90)
border_pen.hideturtle()	


player = turtle.Turtle()
player.color("blue")
player.shape("triangle")
player.penup()
player.speed(0)
player.setposition(0, -250)
player.setheading(90)

playerspeed = 15


number_of_enemies = 5

enemies = []


for i in range(number_of_enemies):
	
	enemies.append(turtle.Turtle())

for enemy in enemies:
	enemy.color("red")
	enemy.shape("circle")
	enemy.penup()
	enemy.speed(0)
	x = random.randint(-200, 200)
	y = random.randint(100, 250)
	enemy.setposition(x, y)

enemyspeed = 2



bullet = turtle.Turtle()
bullet.color("yellow")
bullet.shape("triangle")
bullet.penup()
bullet.speed(0)
bullet.setheading(90)
bullet.shapesize(0.5, 0.5)
bullet.hideturtle()

bulletspeed = 20

bulletstate = "ready"

def fire_bullet():
	
	global bulletstate
	if bulletstate == "ready":
		bulletstate = "fire"
		
		x = player.xcor()
		y = player.ycor() + 10
		bullet.setposition(x, y)
		bullet.showturtle()

def isCollision(t1, t2):
	distance = math.sqrt(math.pow(t1.xcor()-t2.xcor(),2)+math.pow(t1.ycor()-t2.ycor(),2))
	if distance < 15:
		return True
	else:
		return False

while True:
	for enemy in enemies:
		
		x = enemy.xcor()
		x += enemyspeed
		enemy.setx(x)

		
		if enemy.xcor() > 280:
			
			for e in enemies:
				y = e.ycor()
				y -= 40
				e.sety(y)
			
			enemyspeed *= -1
		
		if enemy.xcor() < -280:
			
			for e in enemies:
				y = e.ycor()
				y -= 40
				e.sety(y)
			
			enemyspeed *= -1
			
		
		if isCollision(bullet, enemy):
			
			bullet.hideturtle()
			bulletstate = "ready"
			bullet.setposition(0, -400)
			
			x = random.randint(-200, 200)
			y = random.randint(100, 250)
			enemy.setposition(x, y)
		
		if isCollision(player, enemy):
			player.hideturtle()
			enemy.hideturtle()
			print ("Game Over")
			break

		
	
	if bulletstate == "fire":
		y = bullet.ycor()
		y += bulletspeed
		bullet.sety(y)
	
	
	if bullet.ycor() > 275:
		bullet.hideturtle()
		bulletstate = "ready"
	
	if fileStream and not vs.more():
		break

	
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	
	rects = detector(gray, 0)

	
	for rect in rects:
		
		
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		
		
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		
		ear = (leftEAR + rightEAR) / 2.0

		
		
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		
		
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		
		
		else:
			
			if (COUNTER >= EYE_AR_CONSEC_FRAMES):
						
				
				TOTAL += 1
				fire_bullet()

			
			COUNTER = 0

		
		
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	
	if key == ord("q"):
		break

		
cv2.destroyAllWindows()
vs.stop()
