import pyaudio
import wave
import os
from random import randrange
import subprocess

DATASETPATH = 'rawDataset/'
FINALDATASETPATH = 'speech_dataset/'

numberStraightAnswers = 0

# Creates directories if they do not exist
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getNumber(target):
    num = 0
    targetPath = os.path.join(target, target + '-{}.wav') # example speech_dataset/onion/onion-2.wav
    
    while True:
        if os.path.isfile(os.path.join(DATASETPATH, targetPath.format(num))):
            num += 1
        else:
            print(f'New file {targetPath.format(num)}')
            return num

wantedWords = ['onion', 'pepper', 'tomato']
startingNum = {}

for target in wantedWords:
    check_dir(DATASETPATH + target)
    check_dir(FINALDATASETPATH + target)
    startingNum[target] = getNumber(target)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 2

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
while True:
    target = wantedWords[randrange(0, len(wantedWords))]
    filePath = os.path.join(os.path.join(DATASETPATH, target), target + '-{}.wav').format(startingNum[target])
    startingNum[target] += 1
    print(f"Say {target}")
    print("* recording")

    frames = []

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        
        data = stream.read(CHUNK, exception_on_overflow = False) # , exception_on_overflow = False not working
        frames.append(data)

    print("* done recording")

    # stream.stop_stream()
    
    p.terminate()

    print("Saving in " + filePath)
    wf = wave.open(filePath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    stream.close()

    # Each certain number of answers we will ask users if they want to end
    numberStraightAnswers += 1
    if numberStraightAnswers == 20:
        print("Press enter if you want to continue, press any key + enter to exit")
        if input() == '': # continue
            numberStraightAnswers = 0
        else: # exit
            # execute 1 minute script
            # /tmp/extract_loudest_section/gen/bin/extract_loudest_section $(dirname "${VAR}")'/speech_dataset/onion/*.wav'  $(dirname "${VAR}")/new_onion/
            if not os.path.exists('extract_loudest_section'): # the program needed is not downloaded
                subprocess.call("git clone https://github.com/petewarden/extract_loudest_section", shell=True, check=True)
                subprocess.call('make', cwd="./extract_loudest_section/", shell=True)
            elif not os.path.exists('/tmp/extract_loudest_section/'):
                subprocess.call('make', cwd="./extract_loudest_section/", shell=True)
            for target in wantedWords:
                subprocess.call(f'/tmp/extract_loudest_section/gen/bin/extract_loudest_section $(dirname "${{VAR}}")\'/{DATASETPATH}{target}/*.wav\'  $(dirname "${{VAR}}")/{FINALDATASETPATH}{target}', shell=True)
            break
