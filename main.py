import cv2
import os
import math
import matplotlib.pyplot as plt
import numpy as np

class ImageProcessing:
    def run_threshold():
        image = cv2.imread("./assets/ImageProcessing/rose.png")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bin_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
        _, inv_bin_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
        _, trunc_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TRUNC)
        _, tozero_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TOZERO)
        _, inv_tozero_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TOZERO_INV)
        _, otsu_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_OTSU)

        result_image = [gray_image, bin_thresh, inv_bin_thresh, trunc_thresh, tozero_thresh, inv_tozero_thresh, otsu_thresh]
        result_desc = ["Grayscale", "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV", "OTSU"]

        plt.figure("Threshold Result", figsize=(8, 8))
        for i, (curr_image, curr_desc) in enumerate(zip(result_image, result_desc)):
            plt.subplot(3, 3, (i+1))
            plt.imshow(curr_image, "gray")
            plt.title(curr_desc)
            plt.xticks([])
            plt.yticks([])
        plt.show()
    
    def run_filtering():
        image = cv2.imread("./assets/ImageProcessing/rose.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mean_blur = cv2.blur(image, (11, 11))
        gaussian_blur = cv2.GaussianBlur(image, (11, 11), 5.0)
        median_blur = cv2.medianBlur(image, 11)
        bilateral_blur = cv2.bilateralFilter(image, 5, 150, 150)

        result_image = [image, mean_blur, gaussian_blur, median_blur, bilateral_blur]
        result_desc = ["Original", "Mean", "Gaussian", "Median", "Bilateral"]

        plt.figure("Filtering Result", figsize=(8, 8))
        for i, (curr_image, curr_desc) in enumerate(zip(result_image, result_desc)):
            plt.subplot(3, 3, (i+1))
            plt.imshow(curr_image)
            plt.title(curr_desc)
            plt.xticks([])
            plt.yticks([])
        plt.show()

class EdgeDetection:
    def run():
        image = cv2.imread("./assets/EdgeDetection/chess_board.jpeg")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian_uint8 = np.uint8(np.absolute(laplacian))
        
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, 5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, 5)
        
        canny = cv2.Canny(gray_image, 100, 200)

        result_image = [gray_image, laplacian, laplacian_uint8, sobel_x, sobel_y, canny]
        result_desc = ["Original", "Laplacian", "uint8", "Sobel_X", "Sobel_Y", "Canny"]
        plt.figure("Edge Detection Result", figsize=(8, 8))
        for i, (curr_image, curr_desc) in enumerate(zip(result_image, result_desc)):
            plt.subplot(2, 3, (i+1))
            plt.imshow(curr_image, "gray")
            plt.title(curr_desc)
            plt.xticks([])
            plt.yticks([])
        plt.show()

class ShapeDetection:
    def run():
        image_object =  cv2.imread("./assets/ShapeDetection/poster.jpg")
        image_scene = cv2.imread("./assets/ShapeDetection/poster_collection.jpg")

        surf = cv2.SIFT_create()
        
        kp_object, des_object = surf.detectAndCompute(image_object, None)
        kp_scene, des_scene = surf.detectAndCompute(image_scene, None)

        des_object = des_object.astype("f")
        des_scene = des_scene.astype("f")

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des_object, des_scene, 2)

        matchesMask = []
        for i in range(0, len(matches)):
            matchesMask.append([0, 0])

        total_match = 0
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                total_match += 1
                
        img_res = cv2.drawMatchesKnn(
            image_object, kp_object, image_scene,
            kp_scene, matches, None,
            matchColor = [0, 255, 0], 
            singlePointColor = [255, 0, 0], 
            matchesMask = matchesMask
        )
        plt.figure("Shape Detection Result")
        plt.imshow(img_res)
        plt.show()

class PatternRecognition:
    def run():
        train_path = "./assets/PatternRecognition/train"
        person_names = os.listdir(train_path)
        face_cascade = cv2.CascadeClassifier("./assets/haarcascade_frontalface_default.xml")

        face_list = []
        class_list = []
        for index, person_name in enumerate(person_names):
            full_name_path = train_path + "/" + person_name

            for image_path in os.listdir(full_name_path):
                full_image_path = full_name_path + "/" + image_path
                img_gray = cv2.imread(full_image_path, 0)
                detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
                
                if(len(detected_faces) < 1):
                    continue

                for face_rect in detected_faces:
                    x, y, w, h = face_rect
                    face_img = img_gray[y:y+w, x:x+h]

                    face_list.append(face_img)
                    class_list.append(index)

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(face_list, np.array(class_list))

        test_path = "./assets/PatternRecognition/test"
        for image_path in os.listdir(test_path):
            full_image_path = test_path + "/" + image_path
            img_bgr = cv2.imread(full_image_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

            if(len(detected_faces) < 1):
                continue

            for face_rect in detected_faces:
                x, y, w, h = face_rect
                face_img = img_gray[y:y+w, x:x+h]

                res, confidence = face_recognizer.predict(face_img)
                confidence = math.floor(confidence * 100) / 100

                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 1)
                text = person_names[res] + " " + str(confidence) + "%"
                cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
                cv2.imshow("Pattern Recognition Result", img_bgr)
                cv2.waitKey(0)

class FaceDetection:
    def run():
        face_cascade = cv2.CascadeClassifier("./assets/haarcascade_frontalface_default.xml")
        
        img = cv2.imread("./assets/FaceRecognition/paper_rex.jpg")

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 121, 255), 2)
            caught = img[y: y + h, x: x + w]
            caught = cv2.GaussianBlur(caught, (17, 17), 10.0)
            img[y: y + h, x: x + w] = caught
        
        cv2.imshow("Face Detection Result", img)
        cv2.waitKey(0)

class Menu:
    def display():
        print("=================================")
        print("| Qualification Computer Vision |")
        print("=================================")
        print("1. Image Processing")
        print("2. Edge Detection")
        print("3. Shape Detection")
        print("4. Pattern Recognition")
        print("5. Face Detection")
        print("0. Exit")

    def get_index():
        return input(">> ")

    def navigate_to_menu(index):
        if index == "1":
            ImageProcessing.run_threshold()
            ImageProcessing.run_filtering()
        elif index == "2":
            EdgeDetection.run()
        elif index == "3":
            ShapeDetection.run()
        elif index == "4":
            PatternRecognition.run()
        elif index == "5":
            FaceDetection.run()
        elif index == "0":
            print("Thank you for using this program. :D")
            exit(0)

if __name__ == "__main__":
    while True:
        Menu.display()
        index = Menu.get_index()
        Menu.navigate_to_menu(index)
