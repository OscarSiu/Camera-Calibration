Work flow for find_pixel.py

1. Define image path
2. Set GSD (You may calculate the dedicated GSD from excel sheet or GSDcal.py)
3. Define working dis (unchanged by default) 
4. Run the program
5. Left click the mouse for the first point you want to measure
6. Right click the mouse for the second point you want to measure, the distance of two points will be shown on the image
7. Click the middle button of your mouse to reload the original image
8. Measure the distance by going through step 4-6 again
9. Press Enter to end the process and destroy window

Run cameraCalibration.py to obtain camera matrix.
Then run initial_perspective_calibration.py to find GSD 
