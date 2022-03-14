file_string = '''
import imageio
import os
import sys

if len(sys.argv) < 2:
    print("Not enough args - add the full path")

indir = sys.argv[1]
desDir = sys.argv[2]
duration = sys.argv[3]

# Load each file into a list
print("indir", indir)
print("desDir", desDir)
print("duration", duration)
root = os.path.basename(indir)
for current_path1, subDirs, filenames in os.walk(indir):
    basename1 = os.path.basename(current_path1)
    if basename1.startswith( "201" ):
        frames = []
        frames_JPG = []
        for current_path, subDirs, filenames in os.walk(current_path1):       
            print("current_path", current_path)
            for filename in filenames:
                if filename.endswith(".PNG"):
                    print(filename)
                    frames.append(imageio.imread(current_path + "/" + filename))
                if filename.endswith(".JPG"):
                    print(filename)
                    frames_JPG.append(imageio.imread(current_path + "/" + filename))
            basename = os.path.basename(current_path)   
            if len(frames) == 0:
              frames = frames_JPG
            exportname = desDir + "/"+ root + "-" + basename + ".gif"
            exportname2 = desDir + "/" + root + "-" + basename + "-slow.gif"
            kargs = {'duration': duration}
            kargs2 = {'duration': 2}
            imageio.mimsave(exportname, frames, 'GIF', **kargs)
            imageio.mimsave(exportname2, frames, 'GIF', **kargs2)

#  run:  python gifyDir.py D:\DCIM\100APPLE D:\DCIM\gif 0.5

'''