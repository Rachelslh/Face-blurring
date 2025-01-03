import argparse
import glob

from src.model import *


DATA_DIR = "data"
OUTPUT_DIR = "output/retina_face"
SHOW = False

def main(model, preprocess=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_path in sorted(glob.glob(f"{DATA_DIR}/*.jpg")):
        
        if preprocess:
        # Preprocessing, i think this gives a lot of distortions, in case this is enabled, need to adapt the code below to work on all the cubemap's frames
            cube_images_paths = convert_to_cubemap(img_path)

        if model == "yolo":
            # Yolo + Face localization https://github.com/ageitgey/face_recognition?tab=readme-ov-file
            results = detect_objects([img_path], SHOW)
        elif model == "retina_face":
            # RetineFace, https://github.com/serengil/retinaface?tab=readme-ov-file
            results = detect_faces_retina(img_path, SHOW) # works
        elif model == "mtcnn":
            # MTCNN, https://github.com/ipazc/mtcnn?tab=readme-ov-file
            results = detect_faces_mtcnn(img_path, SHOW) # works
        elif model == "deepface":
            # DeepFace, https://github.com/serengil/deepface?tab=readme-ov-file
            results = detect_faces_deepface(img_path, SHOW) # dosen't work for all backends, just mtcnn, fast_mtcnn, retinaFace

        # Postprocessing
        blur(img_path, results, OUTPUT_DIR)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection and blurring")
    parser.add_argument("--model", type=str, default="retina_face", choices=['retina_face', 'deepface', 'mtcnn', 'yolo'], help="Face detection model")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess image")
    args = parser.parse_args()
    
    main(args.model, args.preprocess)
