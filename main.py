import cv2
import numpy as np

# Define HSV ranges for red, yellow, and green
red_lower = np.array([170, 100, 100])
red_upper = np.array([180, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
green_lower = np.array([70, 100, 100])
green_upper = np.array([80, 255, 255])


def track_vehicles(frame, previous_centroids):
  # Convert to HSV for background subtraction
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  background_model = cv2.BackgroundSubtractorMOG2()  # Create background model
  foreground = background_model.apply(hsv)

  # Detect moving objects using foreground mask
  thresh = cv2.threshold(foreground, 25, 255, cv2.THRESH_BINARY)[1]
  kernel = np.ones((5, 5), np.uint8)
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
  closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

  # Find contours and update centroids
  current_centroids = []
  for contour in cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
    if cv2.contourArea(contour) > 500:
      M = cv2.moments(contour)
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      current_centroids.append((cX, cY))

      # Match with previous centroids (replace with Kalman filter for better tracking)
      if previous_centroids is not None:
        for i, (prevX, prevY) in enumerate(previous_centroids):
          dist = ((cX - prevX) ** 2 + (cY - prevY) ** 2) ** 0.5
          if dist < 50:  # Adjust threshold for matching distance
            previous_centroids[i] = (cX, cY)
            break

  return frame, current_centroids


def detect_vehicles(frame):
  """
  Detects vehicles in an image using improved contour analysis and filtering.

  Args:
    frame: The input image (BGR format).

  Returns:
    frame: The image with detected vehicles marked (bounding boxes).
    vehicles: List of (x, y, w, h) coordinates for detected vehicles.
  """

  # Convert to grayscale and apply noise reduction
  grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

  # Adaptive thresholding for better lighting variations
  thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

  # Find contours
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter contours based on area, aspect ratio, and solidity
  vehicles = []
  for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    solidity = cv2.contourArea(contour) / (w * h)

    # Adjust thresholds based on your environment and vehicle sizes
    if 500 < area < 10000 and 0.7 < aspect_ratio < 1.3 and solidity > 0.8:
      vehicles.append((x, y, w, h))
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green bounding box

  return frame, vehicles


def detect_traffic_lights(frame):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Mask for each color
  red_mask = cv2.inRange(hsv, red_lower, red_upper)
  yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
  green_mask = cv2.inRange(hsv, green_lower, green_upper)

  # Find contours for each color
  red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Draw contours and identify dominant color
  for contour in red_contours:
    if cv2.contourArea(contour) > 100:  # Adjust threshold for size
      cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
      light_state = "RED"
      break
  else:
    for contour in yellow_contours:
      if cv2.contourArea(contour) > 100:
        cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
        light_state = "YELLOW"
        break
    else:
      for contour in green_contours:
        
        if cv2.contourArea(contour) > 100:
          cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
          light_state = "GREEN"
          break
      else:
        light_state = "UNKNOWN"

  return frame, light_state

# Example usage
cap = cv2.VideoCapture('./path/video')  # Use your camera index
while True:
  ret, frame = cap.read()
  frame, vehicles = detect_vehicles(frame)
  # frame, light_state = detect_traffic_lights(frame)
  resized_img = cv2.resize(frame, None, fx=0.2, fy=0.2)
  cv2.imshow("Traffic Light Detection", resized_img)
  # print(f"Light state: {light_state}")
  print(f"Detected vehicles: {len(vehicles)}")

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()