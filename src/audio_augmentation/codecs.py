
from io import BytesIO
import random

import torch
import torchaudio
from pydub import AudioSegment
from torch import nn


class Codecs(nn.Module):
    def __init__(self, p=0.2, *args, **kwargs):

        self.codecs = ["opus_16k", "amr", "speex_8khz", "mp3", "opus_8khz", "clear", "clear"]
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, audio):
        target_codec = random.choice(self.codecs)
        print(target_codec)
        if target_codec == "opus_16k":
            return self.encode_decode_opus_16khz(audio)
        if target_codec == "amr":
            return self.encode_decode_amr(audio)
        if target_codec == "mp3":
            return self.encode_decode_mp3(audio)
        if target_codec == "opus_8khz":
            return self.encode_decode_opus_8khz(audio)
        if target_codec == "speex_8khz":
            return self.encode_decode_speex_8khz(audio)
        
        return torchaudio.load(audio)[0]

    @staticmethod
    def encode_decode_opus_16khz(input_file):
        bitrate = random.uniform(6.0, 30.0)  # Random bitrate within range
        audio = AudioSegment.from_file(input_file, format="wav")
        audio = audio.set_frame_rate(16000)
        f = BytesIO()
        audio.export(f, format="opus", bitrate=f"{bitrate}k")

        decoded_audio = AudioSegment.from_file(f, codec="opus")
        decoded_audio = decoded_audio.set_frame_rate(16000)
        decoded_audio.export(f, format="wav")
        aud = torchaudio.load(f)[0]
        return aud
        
    @staticmethod  
    def encode_decode_ulaw(input_file):
        audio = AudioSegment.from_file(input_file, format="wav")
        audio = audio.set_frame_rate(8000)
        f = BytesIO()
        audio.export(f, format="mulaw", bitrate=f"{64}k")
    
        decoded_audio = AudioSegment.from_file(f, format="mulaw")
        decoded_audio.export(f, format="wav")
    
        aud, sr = torchaudio.load(f)
        resampler = torchaudio.transforms.Resample(8000, 16000)
    
        return resampler(aud)

    @staticmethod  
    def encode_decode_g722(input_file):
        audio = AudioSegment.from_file(input_file, format="wav")
        f = BytesIO()
        audio.export(f, format="g722", bitrate=f"{64}k")
    
        decoded_audio = AudioSegment.from_file(f, format="g722")
        decoded_audio.export(f, format="wav")
    
        aud, sr = torchaudio.load(f)
        return aud
        
    @staticmethod
    def encode_decode_gsm(input_file):
        audio = AudioSegment.from_file(input_file, format="wav")
        audio = audio.set_frame_rate(8000)
        f = BytesIO()
        audio.export(f, format="gsm", bitrate=f"{13}k")
    
        decoded_audio = AudioSegment.from_file(f, format="gsm")
        decoded_audio.export(f, format="wav")
    
        aud, sr = torchaudio.load(f)
        f.seak(0)
        resampler = torchaudio.transforms.Resample(8000, 16000)
    
        return resampler(aud)

    @staticmethod  
    def encode_decode_ulaw(input_file):
        audio = AudioSegment.from_file(input_file, format="wav")
        audio = audio.set_frame_rate(8000)
        f = BytesIO()
        audio.export(f, format="alaw", bitrate=f"{64}k")
    
        decoded_audio = AudioSegment.from_file(f, format="alaw")
        decoded_audio.export(f, format="wav")
    
        aud, sr = torchaudio.load(f)
        resampler = torchaudio.transforms.Resample(8000, 16000)
    
        return resampler(aud)

    @staticmethod
    def encode_decode_amr(input_file):
        bitrate = random.uniform(6.6, 23.05)
        audio = AudioSegment.from_file(input_file, format="wav")
        audio = audio.set_frame_rate(8000)
        f = BytesIO()
        audio.export(f, format="amr", bitrate=f"{bitrate}k")

        decoded_audio = AudioSegment.from_file(f, format="amr")
        decoded_audio = decoded_audio.set_frame_rate(16000)
        decoded_audio.export(f, format="wav")
        aud = torchaudio.load(f)[0]
        return aud
    
    @staticmethod
    def encode_decode_mp3(input_file):
        bitrate = random.uniform(45, 256)
        audio = AudioSegment.from_file(input_file, format="wav")
        audio = audio.set_frame_rate(16000)
        f = BytesIO()

        audio.export(f, format="mp3", bitrate=f"{int(bitrate)}k")

        decoded_audio = AudioSegment.from_file(f, format="mp3")
        decoded_audio = decoded_audio.set_frame_rate(16000)
        decoded_audio.export(f, format="wav")
        aud = torchaudio.load(f)[0]
        return aud

    @staticmethod
    def encode_decode_opus_8khz(input_file):
        bitrate = random.uniform(4.0, 20.0)  # Random bitrate within range
        audio = AudioSegment.from_file(input_file, format="wav")
        audio = audio.set_frame_rate(8000)
        f = BytesIO()
        audio.export(f, format="opus", bitrate=f"{bitrate}k")

        decoded_audio = AudioSegment.from_file(f, codec="opus")
        decoded_audio = decoded_audio.set_frame_rate(16000)
        decoded_audio.export(f, format="wav")
        aud = torchaudio.load(f)[0]
        return aud

    @staticmethod
    def encode_decode_speex_8khz(input_file):
        bitrate = random.uniform(3.95, 24.60)
        audio = AudioSegment.from_file(input_file, format="wav")
        audio = audio.set_frame_rate(8000)
        f = BytesIO()
        audio.export(f, format="spx", bitrate=f"{bitrate}k")

        decoded_audio = AudioSegment.from_file(f, codec="speex")
        decoded_audio = decoded_audio.set_frame_rate(16000)
        decoded_audio.export(f, format="wav")
        aud = torchaudio.load(f)[0]
        return aud

if __name__ == '__main__':
    pass