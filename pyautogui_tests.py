# mediapipe_test_webcam_pyautogui.py
# CS 5100 - Fall 2024
# Final Project: Testing various computer control to see if they can be emulated in pyautogui


# Import remote control library
import pyautogui

# Import time library for delays
import time

# Test alt_tab for window switching
# Test 1
#pyautogui.hotkey('alt', 'tab')          # Too fast and only switches 1 application, try manual
# Test 2
pyautogui.keyDown('alt')
for i in range(5):
    pyautogui.keyDown('tab')
    time.sleep(0.5)
    pyautogui.keyUp('tab')
pyautogui.keyUp('alt')