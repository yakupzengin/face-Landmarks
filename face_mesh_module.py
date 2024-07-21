import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticMode = False ,maxFaces = 2 , refineLandmaks= False , minDetectionCon=0.6 , minTrackingCon = 0.):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLandmaks = refineLandmaks
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.staticMode,
            self.maxFaces,
            self.refineLandmaks,
            self.minDetectionCon,
            self.minTrackingCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                face= []
                for id, lm in enumerate(faceLms.landmark):
                    imageHeight, imageWidth, imageChannels = img.shape
                    x, y = int(lm.x * imageWidth), int(lm.y * imageHeight)
                    # print(f"id : {id} , x : {x} , y : {y}")
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #             0.8, (0, 255, 0), 1)

                    face.append([x,y])
                faces.append(face)
        print(len(faces))
        return img , faces


def main():
    cap = cv2.VideoCapture("dataset/looking-faces.mp4")
    if not cap.isOpened():
        raise FileNotFoundError("The video file cannot be opened.")

    pTime = 0
    detector = FaceMeshDetector(maxFaces=5)
    while True:
        success, img = cap.read()
        img , faces = detector.findFaceMesh(img, True)

        if len(faces) != 0:
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()