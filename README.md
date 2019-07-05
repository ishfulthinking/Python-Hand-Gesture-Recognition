# Hand gesture recognition in Python using OpenCV
##### TODO: Add photos for each "checkpoint"

| ![Pointing](media/pointing.gif)
| ![Scissors](media/scissors.gif)
| ![Waving](media/waving.gif) |  

This guide will teach you how to code a computer vision program that recognizes simple hand gestures:
- Waving
- Pointing (one finger extended)
- Scissors (two fingers extended)
- Rock (no fingers extended)

The easiest way to get this running is to use a Jupyter Notebook, which allows you to write your Python 
code in modules and run each individually or as a group.  
  


## Pre-coding: Get acquainted and plan ahead
Before we do any coding, it's important to think of how we want to approach the task, especially 
because there are multiple ways to code a computer vision program like this one.  

To focus on the user's hand, we can use [background subtraction in OpenCV](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html): basically, we first take a snapshot of the user's background, then we compare all subsequent 
frames to the snapshot and check the difference. Using thresholding, we can highlight the difference between the two images 
to find the object of interest (in this case, the user's hand).  

To make the thresholding easier, we can also focus on only a portion of the window, which will help with both user comfort 
(they won't have to stay off-screen to use the program) and runtime (since less pixels will be checked for gestures).  

For recognizing gestures, I found the simplest strategy is to have a "gesture recognizer" with two outputs of different priority:  
- one that checks how much the center of the hand is moving and how quickly, to check for waving (high priority)  
- one that counts the number of fingers extended, then selects the gesture based on the number (low priority)  

The priority aspect is important since, if the user's hand is waving quickly, we shouldn't waste time trying to 
count the number of fingers extended (especially since they'll likely be too blurry to count reliably anyway).  
  
  
  
## Objective 1: Write the base for the program (take input, show region of interest as a square)  
First things first: we have to lay out the structure for our program and create the tools that we'll use later on. 
For example, we know we're gonna hold the background as a variable, and we could make holding hand data easier by 
storing it in variables contained inside an object.  We also have to import the libraries we're going to use, too!  

### Step 1a: Import libraries and create global variables  
Let's get started! We first have to import the Numpy and OpenCV libraries.  
We also want to declare some global variables that'll be used between multiple functions, such as 
the background, calibration settings, and more.   

```python
import numpy as np
import cv2

# Hold the background frame for background subtraction.
background = None
# Hold the hand's data so all its details are in one place.
hand = None
# Variables to count how many frames have passed and to set the size of the window.
frames_elapsed = 0
FRAME_HEIGHT = 200
FRAME_WIDTH = 300
# Humans come in a ton of beautiful shades and colors.
# Try editing these if your program has trouble recognizing your skin tone.
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18
```

### Step 1b: Write a loop to get frames from the camera while the program runs  
Before we can get into the background subtraction, thresholding, and more, we have to write code 
so that the camera can actually take input for processing.  

Create a new cell for the main function. To get input from your system's camera, we use cv2's VideoCapture function.  
(Note: If it ends up using the wrong camera, try replacing 0 with 1, 2, etc. For me, my self-facing camera is 1)  
```python
capture = cv2.VideoCapture(0)
```

And then write a loop immediately after it to actually read frames from the camera constantly, until the user presses x to exit:  
```python
while (True):
    # Store the frame from the video capture and resize it to the window size.
    ret, frame = capture.read()
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Show the previously captured frame.
    cv2.imshow("Camera Input", frame)
    # Check if user wants to exit.
    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break

# When we exit the loop, we have to stop the capture too.
capture.release()
cv2.destroyAllWindows()
```

Great, now our program can take input from the camera! But you might notice it doesn't work like a mirror, which 
makes it confusing to use. Let's add these 2 lines right after the line beginning with ```frame = ```:  

```python
    # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.
    frame = cv2.flip(frame, 1)
```  

Your program should now work like a mirror. Perfect!  

### Step 1c: Partition the region of interest  
Now we have to detect the user's hand. Let's first set some values for the region of interest and frame count.  
Within the main function, before even taking the capture of the screen, type this:  

```python
# Our region of interest will be the top right part of the frame.
region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH / 2)
region_right = FRAME_WIDTH

frames_elapsed = 0
```
(Note: 0,0 is the top left pixel of the frame, and values increase as we move away from that corner.)

As said before, partitioning which part of the screen a hand should be in is very useful for cutting down 
on runtime (less pixels to check for a hand) and finding where the hand is on the screen.  

To make use of the program easier, we'll draw a rectangle on-screen to show where the user should put 
their hand. We have to make sure that we add it to only a copy of the frame, since we'll need the frame intact 
for partitioning the hand gesture.  

Add these lines after the line where we flip the image:  

```python
# Create a copy of the current frame solely for display, not gesture recognition.
display_frame = frame.copy()

# Draw a rectangle on-screen to show where the user should put their hand.
cv2.rectangle(display_frame, (region_left, region_top), (region_right, region_bottom), (255,255,255), 2)
```

Since we're now using display_frame, make sure you edit your cv2.imshow() call from before to use 
display_frame as a parameter, not frame anymore, and increment frames_elapsed:  

```python
cv2.imshow("Camera Input", display_frame)
frames_elapsed += 1
```

So now we have a program that will take input from the camera and return the frames with 
a square drawn where the user should put their hand. Check out [checkpoint 1 to make sure you've got it right](/checkpoint1.ipynb).
  
  
  
## Objective 2: Create tools to hold hand data and write on screen  

Now that we have our base input code working, let's continue building our foundation. Before we 
jump into coding the background differencing, finger counting, etc. let's start organized so we 
don't have to clean up a ton later.  
It would be disorganized to have a bunch of global variables for all the data of the hand, so we'll 
create an object class to hold all that data and update it. It would also be useless to have our 
gesture recognition functions coded without being able to print the results to the screen, so let's 
do that too.  

### Step 2a: Create an object class to hold hand data  

First, create a new cell in the Jupyter notebook. Then code a new object class, HandData, with the 
following variables and constructor:  

```python
class HandData:
    top = (0,0)
    bottom = (0,0)
    left = (0,0)
    right = (0,0)
    centerX = 0
    prevCenterX = 0
    isInFrame = False
    isWaving = False
    fingers = 0
    
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        isInFrame = False
        isWaving = False
```  

We'll also want an "update" function that does the same as above without creating a new object.  

```python
    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
```
(You might notice that we don't update the variables related to the center of the hand nor finger count; 
we'll do that later when we code to detect hand waving and finger count recognition.)  

### Step 2b: Create a function that writes hand data on the screen  

We want to tell the user if the background is being calibrated, if the hand isn't in the frame, 
how many fingers they're holding up, etc. so let's create a helper function to do exactly that.  

What are the things we need it to say?  
- "Calibrating..." if less than CALIBRATION_TIME frames have elapsed since starting the program
- "Hand not found" if a hand isn't in the region of interest
- "Waving" if the user is waving their hand
- The name of the gesture if they're holding fingers up  

So let's create a new cell in the notebook and write the function we need:  

```python
# Here we take the current frame, the number of frames elapsed, and how many fingers we've detected
# so we can print on the screen which gesture is happening (or if the camera is calibrating).
def write_on_image(frame, hand):
    text = "Searching..."

    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."
    elif hand == None or hand.isInFrame == False:
        text = "No hand detected"
    else:
        if hand.isWaving:
            text = "Waving"
        elif hand.fingers == 0:
            text = "Rock"
        elif hand.fingers == 1:
            text = "Pointing"
        elif hand.fingers == 2:
            text = "Scissors"
    
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0  ,  0,  0),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)
```

Then, add the helper function to the main function so it can be called, just before the line with cv2.imshow():    

```python
... # Write the action the hand is doing on the screen.
    write_on_image(display_frame, hand)
```

Now the application will be more organized later on (less headaches!) and we can check the 
status of the gesture recognizer using the write_on_image function.  
  
  
  
## Objective 3: Recognize when a hand is in the region of interest  

Let's use the background differencing concept from before to notice when a hand is in the region 
of interest. We need a background that we save at the beginning (which is based on the number of 
frames elapsed since starting the program), then a function that can separate the background from 
subsequent frames.  

### Step 3a: Get the background ready for averaging  

If you're totally new to computer vision, there's value in [reading about edge detection as a principle](https://www.mathworks.com/discovery/edge-detection.html) -- it explains some of the practices we're about to implement. But if you want 
the summarized notes, here they are:  
- Edge detection is easiest by noting sudden differences in brightness/lighting  
- As a result, it's best to convert the frame into a black & white image before processing it  
- To avoid stray pixels (image noise) being labeled as edges, we can smoothen the image with a Gaussian blur  

So let's grab the background, turn it gray, smoothen it a bit, THEN pass it into a function that 
accumulates the averages of the background.  

In the main function's while loop, just after grabbing the frame through cv2.resize(), clone the frame so 
we can edit it without editing the output frame that we'll show the user. Then we isolate the region of 
interest from the rest of the frame:

```python
... frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # Clone the resized input frame, so we can edit it without editing the display frame.
    clone = frame.copy()

    # Separate the region of interest from the rest of the frame.
    region = clone[region_top:region_bottom, region_left:region_right]
    # Make it grayscale so we can detect the edges more easily.
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Use a Gaussian blur to prevent image noise from being labeled as an edge.
    gray = cv2.GaussianBlur(gray, (7,7), 0)
...
```

### Step 3b: Average the first frames of input to get the background

If we only take the first frame of background, we'll immediately run into issues due to lighting 
changes. If your webcam is anything like mine, it spends its first several frames adjusting for 
shadows, which is not what we want for our program.  

Remember our frames_elapsed and CALIBRATION_TIME variables from before? This is where they come in handy! 
I've found that averaging the first ~30 frames of camera input is enough to overcome that obstacle of initial 
lighting adjustments.  

So let's go to the main function and, after we do the Gaussian blur, put an if statement that 
checks if 30 frames have passed since the program first began running:  

```python
... gray = cv2.GaussianBlur(gray, (7,7), 0)

    if frames_elapsed < CALIBRATION_TIME:
        get_avg(gray)
    
    frames_elapsed += 1

...
```

Then, in a new cell in the Jupyter notebook, write the function get_avg():  

```python
# Get the average background of the first CALIBRATION_TIME frames.
# We use a weighted average of these frames to generate the background.
def get_avg(image):
    if background is None:
        background = image.copy().astype("float")
        return
    
    cv2.accumulateWeighted(image, background, BG_WEIGHT)
```

### Step 3c: Use differencing to isolate a hand from the background  

Next, we can write a function to segment the image and mark which parts of the region of interest 
are covered by a hand. Create a new cell in the Jupyter notebook and create the function segment().  

To segment the image properly, we have to follow these steps:
- Get the absolute difference between the current frame and the previous averages of the background.
- Threshold that difference, so the results are binary: either it's part of the background, or it isn't.
- Get the [contours](https://datacarpentry.org/image-processing/09-contours/) of the shape we thresholded. OpenCV will do this for us and return an outline of the shape.

We can see that in code form here:  

```python
# Here we use differencing to separate the background from the object of interest.
def segment(image):
    # Find the absolute difference between the background and the current frame.
    diff = cv2.absdiff(background.astype(np.uint8), image)

    # Threshold that image with a strict 0 or 1 ruling so only the foreground remains.
    thresholded_image = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # Get the contours of the image, which will return an outline of the hand.
    (_, contours, _) = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If we didn't get anything, there's no hand.
    if len(contours) == 0:
        if handData is not None:
            handData.isInFrame = False
        return
    # Otherwise return a tuple of the filled hand (thresholded_image), along with the outline (segmented_image).
    else:
        if handData is not None:
            handData.isInFrame = True
        segmented_image = max(contours, key = cv2.contourArea)
        return (thresholded_image, segmented_image)
```

### Step 3d: Incorporate the background differencing into our main loop  

After we've gotten the average of the first CALIBRATION_TIME frames, we can segment the gray version of the 
frame. Added these two lines to the previous if statement that calls get_avg():

```python
... if frames_elapsed < CALIBRATION_TIME:
        get_avg(gray)
    else:
        (thresholded_image, segmented_image) = segment(gray)
        if (thresholded_image, segmented_image) is not None:
            getHandData(thresholded, segmented)
```

All together, our main function now looks like this:  

```python
# Main function: Get feed from camera #

# Our region of interest will be the top right part of the frame.
region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = 0
region_right = int(FRAME_WIDTH / 2)

frames_elapsed = 0

capture = cv2.VideoCapture(0)

while (True):
    # Store the frame from the video capture and resize it to the window size.
    ret, frame = capture.read()
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # Clone the resized input frame, so we can edit it without editing the display frame.
    clone = frame.copy()

    # Separate the region of interest from the rest of the frame.
    region = clone[region_top:region_bottom, region_left:region_right]
    # Make it grayscale so we can detect the edges more easily.
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Use a Gaussian blur to prevent image noise from being labeled as an edge.
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    
    if frames_elapsed < CALIBRATION_TIME:
        get_avg(gray)
    else:
        (thresholded_image, segmented_image) = segment(gray)
        if (thresholded_image, segmented_image) is not None:
            getHandData(thresholded, segmented)
    
    frames_elapsed += 1

    # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.
    display_frame = cv2.flip(frame, 1)

    # Draw a rectangle on-screen to show where the user should put their hand.
    cv2.rectangle(display_frame, (region_right, region_top), (FRAME_WIDTH, region_bottom), (255,255,255), 2)

    # Show the previously captured frame.
    write_on_image(display_frame, handData)
    cv2.imshow('Camera Input', display_frame)
    # Check if user wants to exit.
    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break

# When we exit the loop, we have to stop the capture too.
capture.release()
cv2.destroyAllWindows()
```  





## Objective 4: Recognize when the user waves  

The first gesture we can get our program to recognize is waving -- it's easier than counting 
fingers and also has higher priority compared to deciphering finger-based gestures.  

How do we do that? Now that we have the image segmentation function, we can get an isolated image
of the hand. We take that isolated hand shape and find its highest and lowest x and y values so 
we can find the center, and if the center point is moving a lot in a short amount of time, the 
user is waving!  

### Step 4a: Program the system to get the hand's center point  

To reduce the amount of math we have to do, we can first create a [convex hull](https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/) of the segmented image. 
That'll "crop" the image so just the hand is there, not the hand + empty space surrounding it.  

The left side of the hand is the lowest x value equal to 1 in the segmented image, while the right 
side is the highest, the top is the lowest y value, and the bottom is the highest.  

Let's create a new function that gets the hand's dimensions, center, etc. so we can use this info 
both for detecting waving and for counting fingers later on.  

```python
def getHandData(thresholded_img, segmented_img):
    # Enclose the area around the extremities in a convex hull to connect all outcroppings.
    convexHull = cv2.convexHull(segmented)
    
    # Find the extremities for the convex hull and store them as points.
    top    = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left   = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right  = tuple(convexHull[convexHull[:, :, 0].argmax()][0])
    
    # Get the center of the palm, so we can check for waving and find the fingers.
    centerX = int((left[0] + right[0]) / 2)
```  

We will call the constructor if the object is null, and update its data if it already exists:  

```python
    # We put all the info into an object for handy extraction (get it? handy?)
    if handData == None:
        handData = Hand(top, bottom, left, right, centerX)
    else:
        handData.update(top, bottom, left, right)
```  

### Step 4b: Create a function for the handData object to check for waving  

It would be difficult to check for waving every frame, as that would require the user to wave 
their hand VERY quickly every single frame. Instead, let's check every fourth frame if the 
center of the user hand has moved significantly, since that would be a better indicator of waving.  

In the getHandData() function we just created, append this:  

```python
    if frames_elapsed % 4 == 0:
        handData.checkForWaving(centerX)
```  

Within the Hand class itself, create the function checkForWaving(). It'll update the current centerX 
and previous centerX of the hand, then check if they differ enough to signify waving:  

```python
    def checkForWaving(self, centerX):
        self.prevCenterX = self.centerX
        self.centerX = centerX
        
        if abs(handData.centerX - handData.prevCenterX > 3):
            handData.isWaving = True
        else:
            handData.isWaving = False
```  

If all has gone according to plan, your program should now recognize waving!  