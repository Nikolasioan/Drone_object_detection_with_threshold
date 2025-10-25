from utils import generate_image, object_detection, extract_bounding_boxes, Direction

def main(number_of_images):
    #generate_image(number_of_images=number_of_images)
    object_detection(number_of_images=number_of_images)
    boxes_array, H, W = extract_bounding_boxes(number_of_images=number_of_images)
    Direction(boxes_array, H, W, number_of_images=number_of_images)

if __name__ == "__main__":
    main(number_of_images=5)
