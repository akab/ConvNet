import os


def video2images(src, train_path=r'data\images', test_path=r'data\test_images', factor=2):
    """
    Extracts all frames from a video and saves them as jpg
    https://github.com/thatbrguy/Pedestrian-Detection/blob/master/extract_towncentre.py
    :param src:
    :param train_path:
    :param test_path:
    :param factor:
    :return:
    """

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    # frame = 0
    # cap = cv2.VideoCapture(src)
    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print('Total Frame Count:', length)
    #
    # while True:
    #     check, img = cap.read()
    #     if check:
    #         if frame < 3600:
    #             path = train_path
    #         else:
    #             path = test_path
    #
    #         img = cv2.resize(img, (1920 // factor, 1080 // factor))
    #         cv2.imwrite(os.path.join(path, str(frame) + ".jpg"), img)
    #
    #         frame += 1
    #         print('Processed: ', frame, end='\r')
    #     else:
    #         break
    # cap.release()
