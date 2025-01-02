import os

import cv2
import matplotlib.pyplot as plt


def convert_to_cubemap(path):
    # create directory
    os.makedirs("data/cubemaps", exist_ok=True)
    import py360convert
    # Load the equirectangular image
    img_equirectangular = cv2.imread(path)
    '''
    # Convert to cubemap with individual faces
    cube_faces = py360convert.e2c(img_equirectangular, face_w=512, cube_format="list")

    # Save each cube face
    face_names = ["front", "right", "back", "left", "top", "bottom"]
    for i, face in enumerate(cube_faces):
        cv2.imwrite(f"data/cubemaps/{path.split('.')[0]}_cubemap_{face_names[i]}.jpg", face)
            
    # Arrange in a 3x2 grid layout
    top_row = np.hstack(cube_faces[:3])  # Front, Right, Back
    bottom_row = np.hstack(cube_faces[3:])  # Left, Top, Bottom
    cubemap = np.vstack([top_row, bottom_row])
    '''
    # Convert to cubemap with individual faces
    cubemap = py360convert.e2c(img_equirectangular, face_w=512)
    
    # Display the cubemap layout
    cv2.imshow(f"Cubemap Layout", cubemap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # save img
    filename = path.split('/')[-1]
    path = f"data/cubemaps/{filename.split('.')[0]}_cubemap.jpg"
    cv2.imwrite(path, cubemap)
    
    return path


def detect_objects(images_path, show=False):
    from ultralytics import YOLO
    import face_recognition
    # Load a model
    model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
    # Run batched inference on a list of images
    result = model(images_path)[0]  # return a list of Results objects

    # Process result list
    boxes = result.boxes.cpu()  # Boxes object for bounding box outputs
    probs = result.probs  # Probs object for classification outputs
    
    result.show()  # display to screen
    #result.save(filename="result.jpg")  # save to disk
    
    # Get bounding box coordinates
    image_crops = []
    faces_bboxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxyn.numpy()[0].tolist()  
        x1, y1, x2, y2 = denormalize_yolo_boxes((x1, y1, x2, y2), result.orig_shape[1], result.orig_shape[0])
        img_crop = result.orig_img[y1:y2, x1:x2]
        image_crops.append(img_crop)
        face_locations = face_recognition.face_locations(img_crop)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            x1_face, y1_face, x2_face, y2_face = x1 + left, y1 + top, x1 + right, y1 + bottom
            faces_bboxes.append((x1_face, y1_face, x2_face, y2_face))
            face_image = img_crop[top:bottom, left:right]
            
            if show:
                cv2.imshow("Face", face_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
    
    return faces_bboxes
               
    
def detect_faces_retina(image_path, show=False):
    from retinaface import RetinaFace
    
    resp = RetinaFace.detect_faces(image_path)
    faces_bboxes = [d['facial_area'] for d in resp.values()]
    # For plotting, let's just extract directly from RetinaFace
    faces = RetinaFace.extract_faces(img_path = image_path, align = True)
    
    if show:
        for face in faces:
            plt.imshow(face)
            plt.show()
             
    return faces_bboxes

             
def detect_faces_mtcnn(image_path, show=False):
    from mtcnn import MTCNN
    from mtcnn.utils.images import load_image
    from mtcnn.utils.plotting import plot
    
    # Create a detector instance
    detector = MTCNN(device="CPU:0")

    # Load an image
    image = load_image(image_path)

    # Detect faces in the image
    result = detector.detect_faces(image)
    faces_bboxes = [res['box'] for res in result]
    faces_bboxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces_bboxes]
    
    if show:
        for res in result:
            plt.figure()
            plt.imshow(plot(image, res))
            plt.show()

    return faces_bboxes


def detect_faces_deepface(image_path, show=False):
    from deepface import DeepFace
    
    faces_bboxes = []
    try:
        face_objs = DeepFace.extract_faces(
            img_path = image_path, 
            detector_backend = 'fastmtcnn',
            align = True,
            )
        
        for face in face_objs:
            facial_area = face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            faces_bboxes.append([x, y, x + w, y + h])
            
            if show:
                cv2.imshow("Face", face['face'])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
    except ValueError as e:
        print(f"Error: {e}")
        
    return faces_bboxes
   
            
def blur(image_path, bboxes, output_dir):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return
    
    filename = image_path.split('/')[-1].split('.')[0]
    image = cv2.imread(image_path)
    for bbox in bboxes:
        blurred_image = cv2.blur(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (33, 33))
        image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = blurred_image
        
    cv2.imwrite(f"{output_dir}/{filename}_processed.jpg", image)


def denormalize_yolo_boxes(coords, image_width, image_height):
    """
    Denormalize YOLO bounding boxes from normalized coordinates to pixel coordinates.

    Args:
        normalized_boxes: List of normalized bounding boxes [(center_x, center_y, width, height), ...].
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        List of denormalized bounding boxes [(x_min, y_min, x_max, y_max), ...].
    """
    
    x1, y1, x2, y2 = coords

    # Convert normalized values to pixel values
    x1_pixel = int(x1 * image_width)
    y1_pixel = int(y1 * image_height)
    x2_pixel = int(x2 * image_width)
    y2_pixel = int(y2 * image_height)

    return x1_pixel, y1_pixel, x2_pixel, y2_pixel