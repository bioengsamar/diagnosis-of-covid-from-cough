from pathlib import Path
import os
import librosa
import numpy as np
import pydub


def load_audio_file(file_path):
    input_length = 35278
    data = librosa.core.load(file_path)[0] #, sr=22050
    
    #print(len(data))
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

def Adding_white_noise(data, name):
    wn = np.random.randn(len(data))
    data_wn = data + 0.005*wn
    write('augmentaded_neg_covid/noise/noise_{}'.format(name), 22050, data_wn,normalized=True )
    
def Shifting_sound(data, name):
    data_roll = np.roll(data, 1600)
    write('augmentaded_neg_covid/shift/shift_{}'.format(name),22050, data_roll,normalized=True)
    
    
def stretch_sound(data, rate=1):
    input_length = 35278
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def stretch_data(data, name):
    data_stretch =stretch_sound(data, 1.2)
    write('augmentaded_neg_covid/stretch/stretch_{}'.format(name), 22050, data_stretch, normalized=True)



    
def iterate_files():
    paths = Path("data/neg").glob('**/*.mp3')
    for path in paths:
        path_in_str = str(path)
        Adding_white_noise(load_audio_file(path_in_str), os.path.basename(path_in_str))
        Shifting_sound(load_audio_file(path_in_str), os.path.basename(path_in_str))
        stretch_data(load_audio_file(path_in_str), os.path.basename(path_in_str))
        
if __name__ == "__main__":
    iterate_files()