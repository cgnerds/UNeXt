import cv2
import os

video_folder = 'inputs/test/images'
# 获取图片列表，并排序
frames = os.listdir(video_folder)
# 排序1：按照文件名排序（全数字，位数相同）
frames = sorted(frames)

# 排序2：按照文件名中的数字排序（前缀+数字，位数不同）
# frames.sort(key=lambda x: int(x.split('.')[0][13:]))
# 获取视频中图片的宽高度信息
img = cv2.imread(os.path.join(video_folder, frames[0]))
size = (img.shape[1], img.shape[0])
# 视频格式
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_write = cv2.VideoWriter(os.path.join(video_folder, 'test.mp4'), fourcc, 20, size)
# 写入视频
for frame in frames:
    img = cv2.imread(os.path.join(video_folder, frame))
    video_write.write(img)
video_write.release()

# mp4转码-命令行
# sudo apt-get -y install x264 
# reboot
# ffmpeg -i seg.mp4 -vcodec libx264 -f mp4 output.mp4
# os.system("ffmpeg -i seg.mp4 -vcodec libx264 -f mp4 seg_output.mp4")