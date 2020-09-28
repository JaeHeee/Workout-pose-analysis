import os
import argparse
import cv2
import glob


def crop_and_save(folder_path, save_path):

    label = save_path.split("/")[-1]
    images = glob.glob(f'{folder_path}/*.png')

    for idx, img in enumerate(images):
        src = cv2.imread(img)
        dst = src.copy()
        dst = src[0:350, 200:568] # 사이즈 조절
        resized =  cv2.resize(dst, (432, 368))
        cv2.imwrite(f'{save_path}/{label}_{idx}.png', resized)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f",
                    "--folder_path",
                    type=str,
                    required=True,
                    help="image folder")
    ap.add_argument("-s",
                    "--save_path",
                    type=str,
                    default=os.environ['HOME']+ '/Downloads/',
                    help="save folder")
    args = vars(ap.parse_args())
    print(args['folder_path'], args['save_path'])
    crop_and_save(args['folder_path'], args['save_path'])