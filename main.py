import cv2
import os
import mediapipe as mp
import argparse


def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur faces
            img[y1:y1 + h, x1:x1 +
                w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (50, 50))

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=['image', 'video', 'webcam'], default='webcam', help="Select input mode")
    parser.add_argument("--filePath", type=str,
                        help="Path to input file (image or video)")
    args = parser.parse_args()

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    mp_face_detection = mp.solutions.face_detection  # type: ignore

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if args.mode == 'image':
            img = cv2.imread(args.filePath)
            img = process_img(img, face_detection)
            cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

        elif args.mode == 'video':
            cap = cv2.VideoCapture(args.filePath)
            ret, frame = cap.read()

            output_video = cv2.VideoWriter(
                os.path.join(output_dir, 'output.mp4'),
                cv2.VideoWriter_fourcc(*'MP4V'),  # type: ignore
                30,
                (frame.shape[1], frame.shape[0])
            )

            while ret:
                frame = process_img(frame, face_detection)
                output_video.write(frame)
                ret, frame = cap.read()

            cap.release()
            output_video.release()
            cv2.destroyAllWindows()

        elif args.mode == 'webcam':
            cap = cv2.VideoCapture(0)
            # Fallback in case fps returns 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            delay = int(1000 / fps)

            ret, frame = cap.read()
            while ret:
                frame = process_img(frame, face_detection)
                cv2.imshow('Webcam Face Blur', frame)

                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break

                ret, frame = cap.read()

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
