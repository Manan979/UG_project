import torch
import cv2
import math
import csv
import numpy as np

### Reading data from SEM image scale bar ####

# Function to calculate distance between two points
scalebar = 1
def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Load the image
image = cv2.imread('./new_test_data/IMG-20230406-WA0010.jpg')

# Display the image and use cv2.setMouseCallback() to select the endpoints of the scale bar
points = []

def click_event(event, x, y, flags, params):
    global scalebar
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        if len(points) == 2:
            cv2.line(image, points[0], points[1], (255, 0, 0), 2)
            length = calculate_distance(points[0], points[1])
            scalebar = length
            print(f'Pixel length of scale bar: {length}')
        cv2.imshow('Image', image)



cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
### Reading from scalebar Completed ###


# Load the YOLOv5 model
model = torch.hub.load(
    'ultralytics/yolov5', 'custom',
    './models/1300_images.pt', force_reload=True
)

# Read the image
name = 'IMG-20230406-WA0010.jpg'
im = './new_test_data/' + name
img = cv2.imread(im)

# Run the model on the image
results = model(im)
results.show()

# Process YOLOv5 results
useful_results = results.xyxy[0].numpy()
for result in useful_results:
    xmin, ymin, xmax, ymax, confidence, clas = result
    if confidence > 0.6:  # Check if confidence is greater than 60%
        x_centre = math.floor((xmin + xmax) / 2)
        y_centre = math.floor((ymin + ymax) / 2)
        radius = math.floor(min((xmax - xmin - 10) / 2, (ymax - ymin - 10) / 2))
        cv2.circle(img, (x_centre, y_centre), radius, (0, 255, 0), 2)

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur with a larger kernel to reduce noise
blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)

# Apply adaptive thresholding for a cleaner binary image
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Use morphological operations to remove small objects or noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Detect edges
edges = cv2.Canny(opened, 100, 200)

# Find contours using only the external contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Define a function to calculate circularity
def circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return False
    circularity_value = 4 * math.pi * (area / (perimeter ** 2))
    return circularity_value


# Filter contours by circularity
min_circularity = 0.7  # Adjust this value as needed (1 is a perfect circle)
circular_contours = [cnt for cnt in contours if circularity(cnt) > min_circularity]

# Draw borders, calculate area, and number the circular contours
for i, contour in enumerate(circular_contours):
    # Draw the contour
    cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)  # Use a different color for contour detection

    # Calculate the area
    area = cv2.contourArea(contour)
    print(f'Contour #{i + 1} Area: {area}')

    # Find the bounding rectangle to place the text more accurately
    x, y, w, h = cv2.boundingRect(contour)
    text_position = (x + w // 2, y + h // 2)

    # Put the number on the contour
    cv2.putText(img, f'#{i + 1}: {area:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the image with YOLOv5 detections and numbered circular contours
cv2.imshow('Detections', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Prepare data for CSV output



# Define the minimum and maximum area range
min_area = 100 # Adjust this value as needed
max_area = 10000000  # Adjust this value as needed

# Prepare data for CSV output
csv_data = [['Contour Number', 'Area', 'Diameter']]

# Draw borders, calculate area, diameter, and number the circular contours
for i, contour in enumerate(circular_contours):
    # Calculate the area
    area = cv2.contourArea(contour)

    # Check if the area is within the specified range
    if min_area <= area <= max_area:
        # Calculate the diameter
        diameter = 2 * math.sqrt(area / math.pi)

        # Append data to the CSV data list
        csv_data.append([i + 1, area, diameter])

# Write data to a CSV file
with open('contour_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

# ... [previous code] ...

# Only use is Scale data is available, else comment out below portion

# Known scale from the SEM image (e.g., the scale bar represents 50 micrometers)
actual_length_micrometers = 50   # Change the value as per image scalebar ratio)

# Pixel length of the scale bar measured from the image
pixel_length_scale_bar =  scalebar # Replace with the actual measurement

# Calculate the conversion factor (micrometers per pixel)
conversion_factor = actual_length_micrometers / pixel_length_scale_bar

# Apply the conversion factor to the diameters
for i in range(1, len(csv_data)):
    # Convert diameter from pixels to micrometers
    csv_data[i][2] *= conversion_factor  # Diameter is in the third column

# Write the converted data to a new CSV file
with open('contour_data_micrometers.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)




