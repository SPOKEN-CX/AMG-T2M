import os
import imageio
import cv2
import numpy as np

def gif_to_mp4(gif_path, mp4_path, fps=25):
    gif = imageio.mimread(gif_path, memtest=False)

    if not gif:
        print(f"跳过空文件: {gif_path}")
        return

    height, width = gif[0].shape[0], gif[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    for frame in gif:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

    video.release()
    print(f"已转换: {gif_path} -> {mp4_path}")

def batch_convert_gifs(folder_path, fps=25):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".gif"):
            gif_path = os.path.join(folder_path, filename)
            mp4_filename = os.path.splitext(filename)[0] + ".mp4"
            mp4_path = os.path.join(folder_path, mp4_filename)
            gif_to_mp4(gif_path, mp4_path, fps)

# 示例调用：替换为你自己的文件夹路径
batch_convert_gifs("./user_study", fps=20)
