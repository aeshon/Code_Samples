from imutils.video import VideoStream
import datetime
import argparse
import imutils
import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from tensorflow.lite.python.interpreter import Interpreter as inter
import os
import cv2
import numpy as np
from PIL import Image

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def authorize():
    gauth = GoogleAuth()

    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.credentials is None:
        # Authenticate if they're not there

        # This is what solved the issues:
        gauth.GetFlow()
        gauth.flow.params.update({'access_type': 'offline'})
        gauth.flow.params.update({'approval_prompt': 'force'})

        gauth.LocalWebserverAuth()

    elif gauth.access_token_expired:

        # Refresh them if expired

        gauth.Refresh()
    else:

        # Initialize the saved creds

        gauth.Authorize()

    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)
    return drive


def picture():

    # ec.capture(0, False, "C:/Users/aeshon/Desktop/pic/img.jpg")
    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    # Read picture. ret === True on success
    ret, frame = video_capture.read()
    new_frame = cv2.resize(frame, (320, 240))
    cv2.imwrite("C:/Users/aeshon/Desktop/pic/img.jpg", new_frame)
    # Close device
    video_capture.release()


def object_detector():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())

    # if the video argument is None, then we are reading from webcam
    if args.get("video", None) is None:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

    # otherwise, we are reading from a video file
    else:
        vs = cv2.VideoCapture(args["video"])

    # initialize the first frame in the video stream
    firstFrame = None

    while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        text = "Unoccupied"

        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

            # compute the absolute difference between the current frame and
            # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"
            if text == "Occupied":
                cv2.destroyAllWindows()
                return True

        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    vs.stop() if args.get("video", None) is None else vs.release()
    cv2.destroyAllWindows()


def predict(drive):
    while True:

        if len(os.listdir('C:/Users/aeshon/Desktop/StarlingsFolder')) == 0:
            continue

        model = "C:/Users/aeshon/Downloads/exportedModel.tflite"

        interpreter = inter(model)
        interpreter.allocate_tensors()
        _, height, width, _ = interpreter.get_input_details()[0]['shape']

        for file in os.listdir("C:/Users/aeshon/Desktop/StarlingsFolder"):
            img = Image.open("C:/Users/aeshon/Desktop/StarlingsFolder/" + file).convert('RGB').resize((224, 224), Image.ANTIALIAS)
            results = classify_image(interpreter, img)

            label_id, prob = results[0]
            if label_id == 0:
                print("Animal with " + str(round((prob * 100), 1)) + "% confidence")
            else:
                model = "C:/Users/aeshon/Downloads/exportedModelBVS.tflite"

                interpreter = inter(model)
                interpreter.allocate_tensors()
                _, height, width, _ = interpreter.get_input_details()[0]['shape']
                results = classify_image(interpreter, img)

                label_id, prob = results[0]

                if label_id == 0:
                    print("Bird with " + str(round((prob * 100), 1)) + "% confidence")
                else:
                    string = "Starling sighted with " + str(round((prob*100), 1)) + "% confidence"
                    write(string)
            os.remove("C:/Users/aeshon/Desktop/StarlingsFolder/" + file)

        upload(drive)


def write(class_result):
    file1 = open("C:/Users/aeshon/Desktop/AlertLog.txt", "a")
    file1.write(str(datetime.datetime.now()) + " " + class_result)
    file1.write("\n")


def upload(drive):
    with open("C:/Users/aeshon/Desktop/AlertLog.txt", "r") as file:
        upload_file_to_specific_folder(drive=drive, file=file)


def upload_file_to_specific_folder(drive, file, folder_id="1Vm0tcB0E3z0rcmhcj8JI92T5vIfpv6hs"):
    file_metadata = {'title': os.path.basename(file.name), "parents": [{"id": folder_id, "kind": "drive#childList"}]}
    folder = drive.CreateFile(file_metadata)
    folder.SetContentFile(file.name)
    folder.Upload()


drive1 = authorize()

if object_detector():
    picture()
predict(drive1)
