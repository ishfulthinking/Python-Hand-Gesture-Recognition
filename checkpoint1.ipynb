{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Python Hand Gesture Recognition (Checkpoint 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This checkpoint is at the end of Objective 1.\n",
    "\n",
    "At this point, your code should be able to get input from the camera and display it on the screen as if it were a mirror. It should also draw a rectangle where the user should put their hand."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Header: Importing libraries and creating global variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import cv2\n",
    "\n",
    "# Hold the background frame for background subtraction.\n",
    "background = None\n",
    "# Hold the hand's data so all its details are in one place.\n",
    "hand = None\n",
    "# Variables to count how many frames have passed and to set the size of the window.\n",
    "frames_elapsed = 0\n",
    "FRAME_HEIGHT = 200\n",
    "FRAME_WIDTH = 300\n",
    "# Humans come in a ton of beautiful shades and colors.\n",
    "# Try editing these if your program has trouble recognizing your skin tone.\n",
    "CALIBRATION_TIME = 30\n",
    "BG_WEIGHT = 0.5\n",
    "OBJ_THRESHOLD = 18"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Main function: Get input from camera and call functions to understand it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Our region of interest will be the top right part of the frame.\n",
    "region_top = 0\n",
    "region_bottom = int(2 * FRAME_HEIGHT / 3)\n",
    "region_left = int(FRAME_WIDTH / 2)\n",
    "region_right = FRAME_WIDTH\n",
    "\n",
    "frames_elapsed = 0\n",
    "\n",
    "capture = cv2.VideoCapture(0)\n",
    "\n",
    "while (True):\n",
    "    # Store the frame from the video capture and resize it to the window size.\n",
    "    ret, frame = capture.read()\n",
    "    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))\n",
    "    # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.\n",
    "    frame = cv2.flip(frame, 1)\n",
    "    \n",
    "    # Show the previously captured frame.\n",
    "    cv2.imshow(\"Camera Input\", frame)\n",
    "    frames_elapsed += 1\n",
    "    # Check if user wants to exit.\n",
    "    if (cv2.waitKey(1) & 0xFF == ord('x')):\n",
    "        break\n",
    "\n",
    "# When we exit the loop, we have to stop the capture too.\n",
    "capture.release()\n",
    "cv2.destroyAllWindows()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
