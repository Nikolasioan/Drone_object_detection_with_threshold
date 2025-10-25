from diffusers import DiffusionPipeline
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image Generation Function
def generate_image(number_of_images=1):

    for i in range(0, number_of_images):

        pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-turbo") # That is the model’s website on Hugging Face: https://huggingface.co/stabilityai/sd-turbo

        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu") # Use GPU only if available (Exclusively NVIDIA GPU). If not, use CPU.

        # Short, precise prompts tend to produce better results.
        # Avoid ambiguous words, as they can confuse the model.
        prompt = "Sky and reflecting sea. Objects like buoys, debris etc. exclusively floating inside the water. Realistic, high quality."

        image = pipe(prompt).images[0]

        image.save(f"./Images/Original/output_{i+1}.png")

# Threshold Object Detection Function
def object_detection(number_of_images=1):

    for i in range(0, number_of_images):

        img = cv2.imread(f"./Images/Original/output_{i+1}.png") # Load image.

        #Convert image to HSV: H = Colour, S = Saturation, V = Brightness
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(HSV)

        # Define blue colour range in HSV
        lower_blue = np.array([90, 60, 40])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(HSV, lower_blue, upper_blue) # Mask for blue areas

        # Bright/White  areas (clouds/glare/sun) within blue regions # Define Bright/White (Cloud, glare, sun) areas
        Bright_White_mask = ((S < 60) & (V > 180)).astype(np.uint8) * 255

        mask = cv2.bitwise_or(blue_mask, Bright_White_mask) # Combine blue areas and bright/white areas. The result is a binary mask.

        # Reflections 
        #Reflections_mask = (S > 245).astype(np.uint8) * 255
        #mask = cv2.bitwise_or(mask, Reflections_mask)

        objects_mask = cv2.bitwise_not(mask) # Invert mask to get potential objects.
        
        # Remove noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # 5 x 5 elliptical kernel.
        objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_OPEN, kernel, iterations=1) 
        objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_CLOSE, kernel, iterations=2) 

        cv2.imwrite(f"./Images/Threshold/Threshold_Objects_{i+1}.png", objects_mask)


# Bounding Box Extraction Function
def extract_bounding_boxes(number_of_images=1):

    boxes_array = []
    for i in range(0, number_of_images):

        objects_mask_path = f"./Images/Threshold/Threshold_Objects_{i+1}.png"
        objects_mask = cv2.imread(objects_mask_path, cv2.IMREAD_GRAYSCALE) # Binary mask of objects

        H, W = objects_mask.shape[:2] # Image spatial dimensions

        # num_labels = number of objects + background      labels_info = For each object and for background: [x, y, width, height, area]
        num_labels, _, labels_info, _ = cv2.connectedComponentsWithStats(objects_mask, connectivity=8)

        min_area = max(30, int(0.001 * H * W))   
        boxes = []

        # Remove noise and extract bounding boxes
        for j in range(1, num_labels):  # 0 is background
            x, y, width, height, area = labels_info[j]
            if area < min_area:
                continue
            fill = area / (float(width) * float(height) + 1e-6) 
            if fill < 0.15: 
                continue
            boxes.append((x, y, width, height))

        # Draw bounding boxes on the original image
        img_path = f"./Images/Original/output_{i+1}.png"
        out = cv2.imread(img_path)
        for (x, y, width, height) in boxes:
            cv2.rectangle(out, (x, y), (x + width, y + height), (0, 255, 0), 2)

        cv2.imwrite(f"./Images/Bounding_boxes/Bounding_Boxes_{i+1}.png", out)

        boxes_array.append(boxes)

    return boxes_array, H, W

# Direction of the drone Function
def Direction(boxes_array, H, W, number_of_images=1):
    

    for i in range(0, number_of_images):
        
        target, y_scan, reason = choose_heading(boxes_array[i], W, H)

        origin = (W // 2, int(H * 0.95))  # near bottom center
        tx, ty = target
        vx = tx - origin[0]
        vy = ty - origin[1]

        # Normalization (avoid division by zero)
        norm = (vx**2 + vy**2) ** 0.5
        if norm > 1e-6:
            ux, uy = vx / norm, vy / norm
        else:
            ux, uy = 0.0, -1.0 

        # Visualize scan line, chosen gap center and heading arrow.
        viz = cv2.imread(f"./Images/Bounding_boxes/Bounding_Boxes_{i+1}.png")
        cv2.line(viz, (0, y_scan), (W, y_scan), (255, 255, 0), 1)  # scan line
        cv2.circle(viz, (tx, ty), 5, (0, 255, 255), -1)            # target point
        end_pt = (int(origin[0] + 120 * ux), int(origin[1] + 120 * uy))
        cv2.arrowedLine(viz, origin, end_pt, (255, 0, 0), 3, tipLength=0.18)


        cv2.imwrite(f"./Images/Bounding_boxes_Arrow/Bounding_Boxes_Arrow_{i+1}.png", viz)

        print(f"Chosen heading unit vector: ({ux:.3f}, {uy:.3f}) — reason: {reason}")

# Helper function to choose heading (This function is used inside Direction function)
# start_ratio: where to start scanning (as fraction of H) 
# end_ratio: where to stop scanning (as fraction of H)
# step: how much to lower the scan line each iteration (as fraction of H)
# safety_px: how much horizontal padding to add to each obstacle (For safety reasons)
# min_gap_ratio: minimum required gap width (as fraction of W)
def choose_heading(boxes, W, H, start_ratio=0.55, end_ratio=0.92, step=0.05, safety_px=20, min_gap_ratio=0.12):

    cx = W // 2
    min_gap_px = int(min_gap_ratio * W)

    # If no obstacles, go straight ahead on the first scan line
    if len(boxes) == 0:
        y_scan = int(H * start_ratio)
        return (cx, y_scan), y_scan, "center-clear"

    r = start_ratio
    while r <= end_ratio + 1e-6:
        y_scan = int(H * r)

  
        intervals = []
        for (x, y, w, h) in boxes:
            if y + h >= y_scan:  # obstacle extends into/under the scan line
                left  = max(0, x - safety_px)
                right = min(W, x + w + safety_px)
                intervals.append((left, right))


        intervals.sort()
        merged = []
        for iv in intervals:
            if not merged or iv[0] > merged[-1][1]:
                merged.append([iv[0], iv[1]])
            else:
                merged[-1][1] = max(merged[-1][1], iv[1])

        # Compute gaps as complements of merged intervals in [0, W]
        gaps = []
        cur = 0
        for L, R in merged:
            if L > cur:
                gaps.append((cur, L))
            cur = max(cur, R)
        if cur < W:
            gaps.append((cur, W))


        gaps = [(L, R) for (L, R) in gaps if (R - L) >= min_gap_px]

        if gaps:
            # Prefer going straight if any gap contains the image center
            straight_gap = None
            for (L, R) in gaps:
                if L <= cx <= R:
                    straight_gap = (L, R)
                    break
            if straight_gap is not None:
                tx = (straight_gap[0] + straight_gap[1]) // 2
                return (int(tx), y_scan), y_scan, "center-clear"

            # Otherwise pick the widest gap
            widest = max(gaps, key=lambda g: g[1] - g[0])
            tx = (widest[0] + widest[1]) // 2
            return (int(tx), y_scan), y_scan, "widest-gap"

        # Lower the scan line and try again
        r += step

    # Drone is blocked. No gap found.
    y_scan = int(H * end_ratio)
    return (cx, y_scan), y_scan, "no-gap"