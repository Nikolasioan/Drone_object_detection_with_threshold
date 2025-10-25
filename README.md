# Vision-Based Obstacle Detection and Avoidance over Reflective Water

## Mission
The challenge is to design and simulate a vision-based obstacle detection and avoidance system for a drone flying over reflective water.

## Scenario
Imagine a drone flying low (1–2 meters) over the ocean.  
It must detect and avoid floating obstacles (e.g., buoys or debris) while maintaining stable flight.  

### Computer Vision Simulation
Use Python + OpenCV / PyTorch / TensorFlow to demonstrate a pipeline such as:

- Simulate or generate a few frames of a “water surface” with random floating objects (can be shapes or real images).  
- Use simple detection logic (thresholding, contour detection, optical flow, or YOLO if you prefer).  
- **Output:** draw bounding boxes or trajectories for detected obstacles.

**Primary Limitation of my approach:** The method performs poorly on reflective water. Specifically, it considers reflections to be objects.
