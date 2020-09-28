import pytube
  
def yt_download(video_id, video_path):
    video_url = 'https://youtube/{}'.format(video_id)
    youtube = pytube.YouTube(video_url)
    video = youtube.streams.first()
    video.download(video_path)
    print('youtube video {} saved as: \n\n {}{}.mp4'.format(video_id, video_path, video.title))


if __name__ == '__main__':
    import os
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, required=True,
            help="youtube video id")
    ap.add_argument("-p", "--path", type=str, default=os.environ['HOME']
            + '/Downloads/', help="video download path")
    args = vars(ap.parse_args())
    print(args['video'], args['path'])
    yt_download(args['video'], args['path'])