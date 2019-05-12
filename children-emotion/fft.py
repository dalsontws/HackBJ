import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave
import subprocess

i=0
f,ax = plt.subplots(2)

# Prepare the Plotting Environment with random starting values
x = np.arange(10000)
y = np.random.randn(10000)

# Plot 0 is for raw audio data
li, = ax[0].plot(x, y)
ax[0].set_xlim(0,1000)
ax[0].set_ylim(-5000,5000)
ax[0].set_title("Raw Audio Signal")
# Plot 1 is for the FFT of the audio
li2, = ax[1].plot(x, y)
ax[1].set_xlim(0,5000)
ax[1].set_ylim(-100,100)
ax[1].set_title("Fast Fourier Transform")
# Show the plot, but without blocking updates
plt.pause(0.01)
plt.tight_layout()

FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 1
RATE = 44100
CHUNK = 1024 # 1024bytes of data red from a buffer
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "soundoutput.wav"

wavefile = wave.open(WAVE_OUTPUT_FILENAME, mode='wb')
wavefile.setnchannels(CHANNELS)
wavefile.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
wavefile.setframerate(RATE)

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

global keep_going
keep_going = True
freq = {'max': 0, 'min':1000}
def plot_data(in_data, CHUNK, freq):
    # get and convert the data to float
    audio_data = np.frombuffer(in_data, np.int16)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    data = audio_data * np.hanning(len(audio_data))
    fft = abs(np.fft.fft(data).real)
    freq1 = np.fft.fftfreq(CHUNK)
    freqPeak = freq1[np.where(fft==np.max(fft))]
    freq['max'] = max(freq['max'], freqPeak[0])
    freq['min'] = min(freq['min'], freqPeak[0])
    # print (freqPeak)
    dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))

    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print
    li.set_xdata(np.arange(len(audio_data)))
    li.set_ydata(audio_data)
    li2.set_xdata(np.arange(len(dfft))*10.)
    li2.set_ydata(dfft)

    # Show the updated plot, but without blocking
    plt.pause(0.01)
    if keep_going:
        return True
    else:
        return False

# Open the connection and start streaming the data
stream.start_stream()
frames = []

# Loop so program doesn't end while the stream callback's
# itself for new data
while keep_going:
    if freq['max'] >= 0.145 and freq['min'] <= -0.06:
        applescript = """
        display dialog "CRYING!" ¬
        with title "Alert" ¬
        with icon caution ¬
        buttons {"OK"}
        """

        subprocess.call("osascript -e '{}'".format(applescript), shell=True)
        break
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        plot_data(data, CHUNK, freq)
        #frames.append(np.frombuffer(data, dtype=np.int16))
        wavefile.writeframes(data)

    except KeyboardInterrupt:
        keep_going = False
    except Exception as err:
        print (err)
        print("failed")



# Close up shop (currently not used because KeyboardInterrupt
# is the only way to close)
stream.stop_stream()
stream.close()

audio.terminate()