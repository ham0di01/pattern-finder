import cv2
import numpy as np
import base64
from typing import List, Tuple, Optional
from app.core.config import settings

class VisionService:
    @staticmethod
    def process_image(image_bytes: bytes) -> Tuple[List[float], Optional[str]]:
        # 1. Decode Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        # 2. Resize to fixed width (Standardizes processing)
        target_width = 800
        h, w, _ = img.shape
        scale = target_width / w
        img = cv2.resize(img, (target_width, int(h * scale)))

        # 3. AUTO-CROP: Remove bottom 15% to eliminate Volume Bars
        # (Volume bars usually sit at the bottom and confuse the price line)
        crop_h = int(img.shape[0] * 0.85)
        img = img[:crop_h, :]

        # 4. Convert to HSV (Hue, Saturation, Value) Color Space
        # This allows us to detect "Green" and "Red" regardless of brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 5. Define Color Ranges (These cover standard TradingView colors)
        # Green Mask
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Red Mask (Red wraps around 0 in HSV, so we need two ranges)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        # 6. Combine Masks
        # This image is now Black everywhere EXCEPT where there are candles
        combined_mask = cv2.bitwise_or(mask_green, mask_red)

        # 7. Clean up noise (Morphological Dilation)
        # This thickens thin wicks so they don't get lost
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.dilate(combined_mask, kernel, iterations=2)

        # 8. Extract the Price Line
        points = np.column_stack(np.where(clean_mask > 0))

        if len(points) == 0:
            # Fallback: If no color found (e.g., black/white chart), try adaptive thresholding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clean_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
            points = np.column_stack(np.where(clean_mask > 0))
            if len(points) == 0:
                raise ValueError("No clear chart data found")

        # 9. Map pixels to Time Series (Resample to PATTERN_LENGTH)
        sorted_indices = points[:, 1].argsort()
        points = points[sorted_indices]

        extracted_pattern = []
        chunk_size = img.shape[1] // settings.PATTERN_LENGTH

        # Draw on debug image
        debug_img = img.copy()

        for i in range(settings.PATTERN_LENGTH):
            start_col = i * chunk_size
            end_col = (i + 1) * chunk_size

            strip_points = points[(points[:, 1] >= start_col) & (points[:, 1] < end_col)]

            if len(strip_points) > 0:
                # We take the MEDIAN Y-value to avoid outliers (wicks) throwing off the center
                avg_y = np.median(strip_points[:, 0])
                extracted_pattern.append(-avg_y)  # Invert Y because image coords go down

                # Visual debug
                cv2.circle(debug_img, (int((start_col + end_col) / 2), int(avg_y)), 3, (0, 0, 255), -1)
            else:
                # Copy previous value if gap
                extracted_pattern.append(extracted_pattern[-1] if extracted_pattern else 0)

        # Encode debug image
        _, buffer = cv2.imencode('.png', debug_img)
        debug_base64 = base64.b64encode(buffer).decode('utf-8')

        return extracted_pattern, debug_base64
