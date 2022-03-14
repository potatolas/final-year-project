file_string = '''
import imageio
import os

# Path of video you want to change
clip = os.path.abspath('tu.mp4');


def gifMaker(inputVideoPath, targetFormat):
    outputPath = os.path.splitext(inputVideoPath)[0] + targetFormat

    reader = imageio.get_reader(inputVideoPath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputPath, fps = fps);

    for frames in reader:
        writer.append_data(frames)

    print('DONE!')
    writer.close()


gifMaker(clip,'.gif')
'''