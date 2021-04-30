# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import librosa
import os
from src.instrument_recognition import *
feature_audio = Feature_Audio()
forcast = Forcast()


def read_audio(path):
    if os.path.isfile(path) == False:
        return {"code":"path is not file","data":None,"sr":None}
    signal,sr = librosa.load(path=path, sr=44100)
    if len(signal) / sr > 3:
        signal = signal[:3*sr]
    return {"code":"done","data":signal,"sr":sr}





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    while True:
        print("vui lòng nhập path của file âm thanh:")
        path = input()
        res = read_audio(path)
        if res["code"] != "done":
            print("path is not file")
            print("--------------->^<-------------")
            print('\n\n')
        else:
            print(1)
            signal = res["data"]
            sr = res["sr"]
            x = feature_audio.extract_feature(signal,sr)
            res = forcast.predict(x)
            print(f"file vừa đưa vào là nhạc cụ : {res}")
            print("--------------->^<-------------")
            print('\n\n')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
