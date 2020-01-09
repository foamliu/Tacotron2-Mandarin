# Tacotron 2

A PyTorch implementation of Tacotron2, described in [Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf), an end-to-end text-to-speech(TTS) neural network architecture, which directly converts character text sequence to speech.

## Dataset

[BZNSYP Dataset](https://www.data-baker.com/open_source.html)

## Dependency

- Python 3.6.8
- PyTorch 1.3.0

## Usage
### Data Pre-processing
Extract data:
```bash
$ python extract.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo
Generate mel-spectrogram for text "相对论直接和间接的催生了量子力学的诞生 也为研究微观世界的高速运动确立了全新的数学模型"
```bash
$ python demo.py
```
![image](https://github.com/foamliu/Tacotron2-CN/raw/master/images/mel_spec.jpg)


<a href="audios/sample.wav">audio sample</a>

## 小小的赞助~
<p align="center">
	<img src="https://github.com/foamliu/Tacotron2-Mandarin/blob/master/sponsor.jpg" alt="Sample"  width="324" height="504">
	<p align="center">
		<em>若对您有帮助可给予小小的赞助~</em>
	</p>
</p>
<br/><br/><br/>
