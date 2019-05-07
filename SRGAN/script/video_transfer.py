import cv2
import os
import numpy as np

def video_info(video_name):
    videoCapture = cv2.VideoCapture(video_name)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"video {video_name}: fps: {fps}, frame_total: {frame_numbers}, height {frame_height}, width: {frame_width}")

def video_transfer(src,dst,dst_size):
    print("src info...")
    video_info(src)
    videoCapture = cv2.VideoCapture(src)
    if os.path.exists(dst):
        print(f"{dst} exist and remove now")
        os.remove(dst)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    videoWriter = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, dst_size)
    while (videoCapture.isOpened()):
        ret, frame = videoCapture.read()
        if ret == True:
            img = cv2.resize(frame, dst_size, interpolation=cv2.INTER_CUBIC)
            videoWriter.write(img)
        else:
            break
    videoCapture.release()
    videoWriter.release()
    print(f"finish transfer to {dst}, dst_size {dst_size}")
    print("dst info ...")
    video_info(dst)

def video_merge(src1,src2, dst):
    print("src1 info...")
    video_info(src1)
    print("src2 info...")
    video_info(src2)
    videoCapture1 = cv2.VideoCapture(src1)
    fps1 = videoCapture1.get(cv2.CAP_PROP_FPS)
    src1_width = videoCapture1.get(cv2.CAP_PROP_FRAME_WIDTH)
    src1_height = videoCapture1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    src1_count = videoCapture1.get(cv2.CAP_PROP_FRAME_COUNT)

    videoCapture2 = cv2.VideoCapture(src2)
    fps2 = videoCapture2.get(cv2.CAP_PROP_FPS)
    src2_width = videoCapture2.get(cv2.CAP_PROP_FRAME_WIDTH)
    src2_height = videoCapture2.get(cv2.CAP_PROP_FRAME_HEIGHT)
    src2_count = videoCapture2.get(cv2.CAP_PROP_FRAME_COUNT)

    assert(fps1 == fps2)
    assert(src1_count == src2_count)
    assert(src1_width == src2_width)
    assert(src1_height == src2_height)

    dst_size = (int(src1_width), int(2*src1_height))
    videoWriter = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps1, dst_size)
    while (videoCapture1.isOpened()):
        ret1, frame1 = videoCapture1.read()
        ret2, frame2 = videoCapture2.read()
        if ret1 == True:
            img = np.vstack((frame1,frame2))
            #print(img.shape)
            videoWriter.write(img)
        else:
            break
    videoCapture1.release()
    videoCapture2.release()
    videoWriter.release()
    print("dst info ...")
    video_info(dst)

# video_src_name = "V0111_000162_LML937D3XK0000162.flv"
# video_dst_name = "leon.avi"
# videoCapture = cv2.VideoCapture(video_src_name)
# fps = videoCapture.get(cv2.CAP_PROP_FPS)
# frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
# frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
# frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
# print(f"src video info: fps: {fps}, frame_total: {frame_numbers}, height {frame_height}, width: {frame_width}")
#
# if os.path.exists(video_dst_name):
#     os.remove(video_dst_name)
#
# dst_size = (int(320),int(180))
# videoWriter = cv2.VideoWriter(video_dst_name, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),fps,dst_size)
#
# while(videoCapture.isOpened()):
#     ret, frame = videoCapture.read()
#     if ret == True:
#         img = cv2.resize(frame,dst_size, interpolation = cv2.INTER_CUBIC)
#         # cv2.imshow("test",img)
#         # cv2.waitKey(1000)
#         videoWriter.write(img)
#     else:
#         break
#
# videoCapture.release()
# videoWriter.release()
#
# videoCapture = cv2.VideoCapture(video_dst_name)
# fps = videoCapture.get(cv2.CAP_PROP_FPS)
# frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
# frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
# frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
# print(f"dst video info:  fps: {fps}, frame_total: {frame_numbers}, height {frame_height}, width: {frame_width}")

if __name__ == "__main__":
    #video_transfer("V0111_000162_LML937D3XK0000162.flv","leon.avi",(320,180))
    #video_transfer("V0111_000162_LML937D3XK0000162.flv", "src.avi", (1280, 720))
    #video_transfer("../video/src.avi", "../video/hs.avi", (640, 360))
    #video_transfer("../video/src.avi", "../video/ls.avi", (320,180))
    video_merge("../video/hs.avi","../video/dst.avi","../video/compare.avi")