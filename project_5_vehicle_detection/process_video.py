import cv2
import os.path as path
from SSD import process_frame_bgr_with_SSD, show_SSD_results, get_SSD_model


if __name__ == '__main__':

    SSD_net, bbox_helper, color_palette = get_SSD_model()

    # video_file = 'project_video.mp4'
    video_file = 'C:/Users/minotauro/Google Drive/DEMO_SMARTAREA/modena.mp4'
    # out_path   = 'C:/Users/minotauro/Google Drive/DEMO_SMARTAREA/out_frames'
    out_path = 'C:/temp_frames'

    cap = cv2.VideoCapture(video_file)

    counter = 0
    while True:

        ret, frame = cap.read()

        if ret:
            bboxes = process_frame_bgr_with_SSD(frame, SSD_net, bbox_helper,
                                                min_confidence=0.2,
                                                allow_classes=[2, 7, 14, 15])

            show_SSD_results(bboxes, frame, color_palette=color_palette)

            cv2.imwrite(path.join(out_path, '{:06d}.jpg'.format(counter)), frame)
            # cv2.imshow('', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            counter += 1


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    exit()
