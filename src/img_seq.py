import glob as gb
from shutil import copyfile

emotions_list = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emotions_folders = gb.glob("emotions\\*")  # Returns a list of all folders with participant numbers


def imageWithEmotionEtraction():
    for x in emotions_folders:
        participant = "%s" % x[-4:]  # store current participant number
        for sessions in gb.glob("%s\\*" % x):
            for files in gb.glob("%s\\*" % sessions):
                current_session = files[20:-30]
                file = open(files, 'r')

                emotion = int(float(file.readline()))
                # get path for last image in sequence, which contains the emotion
                sourcefile_emotion = gb.glob("images\\%s\\%s\\*" % (participant, current_session))[-1]
                # do same for neutral image
                sourcefile_neutral = gb.glob("images\\%s\\%s\\*" % (participant, current_session))[0]
                # Generate path to put neutral image
                dest_neut = "selected_set\\neutral\\%s" % sourcefile_neutral[25:]
                # Do same for emotion containing image
                dest_emot = "selected_set\\%s\\%s" % (emotions_list[emotion], sourcefile_emotion[25:])

                copyfile(sourcefile_neutral, dest_neut)  # Copy file
                copyfile(sourcefile_emotion, dest_emot)  # Copy file


if __name__ == '__main__':
    imageWithEmotionEtraction()
    