import cv2
import numpy as np

# Function to calculate distance between two points
def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Load the image
image = cv2.imread('temp.png')

# Display the image and use cv2.setMouseCallback() to select the endpoints of the scale bar
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        if len(points) == 2:
            cv2.line(image, points[0], points[1], (255, 0, 0), 2)
            length = calculate_distance(points[0], points[1])
            print(f'Pixel length of scale bar: {length}')
        cv2.imshow('Image', image)

cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
