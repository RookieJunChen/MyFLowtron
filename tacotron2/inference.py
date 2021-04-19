import matplotlib
import matplotlib.pylab as plt

import IPython.display as ipd

import sys

sys.path.append('waveglow/')
import numpy as np
import torch
import os
import argparse

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write




def plot_data(data, figsize=(16, 4), save_path="test"):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')
    fig.savefig(os.path.join(save_path, 'test{}.png'.format(1)))


def infer(checkpoint_path, waveglow_path, text, save_path):
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    # checkpoint_path = "tacotron2_statedict.pt"
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()


    # waveglow_path = 'waveglow_256channels.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    # denoiser = Denoiser(waveglow)

    # text = "Waveglow is really awesome!"
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T), save_path=save_path)

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet.half(), sigma=0.666)
    # ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
    audio = audio.cpu().numpy()[0]
    # normalize audio for now
    audio = audio / np.abs(audio).max()
    print(audio.shape)

    write(os.path.join(save_path, 'test{}.wav'.format(1)),
          hparams.sampling_rate, audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path',
                        help='Path to flowtron state dict', type=str)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow state dict', type=str)
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-o', "--output_dir", default="results/")
    # parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    infer(args.checkpoint_path, args.waveglow_path, args.text, args.output_dir)
