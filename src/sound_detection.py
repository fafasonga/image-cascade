#!/usr/bin/env python
import time
import pyaudio
from numpy import linspace, short, fromstring, log
from scipy import fft

# Volume Sensitivity, 0.05: Extremely Sensitive, may give false alarms
#             0.1: Probably Ideal volume
#             1: Poorly sensitive, will only go off for relatively loud
SENSITIVITY = 1.0
# Alarm frequencies (Hz) to detect (Use audacity to record a wave and then do Analyze -> Plot Spectrum)
TONE = 1593
# Bandwidth for detection (i.e., detect frequencies within this margin of error of the TONE)
BANDWIDTH = 30
# How many 46ms blips before we declare a beep? (Take the beep length in ms, divide by 46ms, subtract a bit)
beeplength = 1
# How many beeps before we declare an alarm?
alarmlength = 1
# How many false 46ms blips before we declare the alarm is not ringing
resetlength = 10
# How many reset counts until we clear an active alarm?
clearlength = 30
# Enable blip, beep, and reset debug output
debug = False
# Show the most intense frequency detected (useful for configuration)
frequencyoutput = False

# Set up audio sampler -
NUM_SAMPLES = 2048
SAMPLING_RATE = 44100
pa = pyaudio.PyAudio()
_stream = pa.open(format=pyaudio.paInt16,
                  channels=1, rate=SAMPLING_RATE,
                  input=True,
                  frames_per_buffer=NUM_SAMPLES)


def addToLog(appendText):
    print(appendText)
    with open("log.txt", "a") as myfile:
        myfile.write("\n" + appendText)


# ctypes.windll.user32.LockWorkStation()
addToLog("Listening for " + str(alarmlength) + " beeps of " + str(beeplength * 46) + "ms at " + str(TONE) + "Hz")

blipcount = 0
beepcount = 0
resetcount = 0
clearcount = 0
alarm = False

while True:
    while _stream.get_read_available() < NUM_SAMPLES: time.sleep(0.01)
    audio_data = fromstring(_stream.read(
        _stream.get_read_available()), dtype=short)[-NUM_SAMPLES:]
    # Each data point is a signed 16 bit number, so we can normalize by dividing 32*1024
    normalized_data = audio_data / 32768.0
    intensity = abs(fft(normalized_data))[:int(NUM_SAMPLES / 2)]
    frequencies = linspace(0.0, float(SAMPLING_RATE) / 2, num=float(NUM_SAMPLES / 2))
    if frequencyoutput:
        which = intensity[1:].argmax() + 1
        # use quadratic interpolation around the max
        if which != len(intensity) - 1:
            y0, y1, y2 = log(intensity[which - 1:which + 2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            thefreq = (which + x1) * SAMPLING_RATE / NUM_SAMPLES
        else:
            thefreq = which * SAMPLING_RATE / NUM_SAMPLES
        if debug:
            print("\t\t\t\tfreq=", thefreq)
    if max(intensity[(frequencies < TONE + BANDWIDTH) & (frequencies > TONE - BANDWIDTH)]) > max(
            intensity[(frequencies < TONE - 1000) & (frequencies > TONE - 2000)]) + SENSITIVITY:
        blipcount += 1
        resetcount = 0
        if debug:
            print("\t\tBlip", blipcount)
        if blipcount >= beeplength:
            blipcount = 0
            resetcount = 0
            beepcount += 1
            if debug:
                print("\tBeep", beepcount)
            if beepcount >= alarmlength:
                if not alarm:
                    datetime = time.strftime('%Y-%m-%d %H:%M:%S')
                    humantime = time.strftime('%I:%M:%S %p %Z')
                    addToLog("Alarm triggered at " + datetime + " (" + humantime + ")")
                clearcount = 0
                alarm = True
                if debug:
                    print("Alarm!")
                beepcount = 0
    else:
        blipcount = 0
        resetcount += 1
        if debug:
            print("\t\t\treset", resetcount)
        if resetcount >= resetlength:
            resetcount = 0
            beepcount = 0
            if alarm:
                clearcount += 1
                if debug:
                    print("\t\tclear", clearcount)
                if clearcount >= clearlength:
                    clearcount = 0
                    addToLog("Listening...")
                    alarm = False
            else:
                if debug:
                    print("No alarm")
    time.sleep(0.01)
