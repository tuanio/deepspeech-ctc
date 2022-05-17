# Deep Speech with CTC Loss

## Introduction

Deep Speech model is one of the ASR that got the SOTA in Speech Recognition domain. In this respository, I use Deep Speech with Vivos Vietnamese Dataset. 

## How to use this respository

1. Clone this project to current directory. Using those commands:
```
!git init
!git remote add origin https://github.com/tuanio/deepspeech-ctc
!git pull origin main
```
2. Install requirement packages
```
!pip install -r requirements.txt
```

Then install `ctcdecode` from this respository: https://github.com/parlance/ctcdecode

3. Edit `configs.yaml` file for appropriation.
4. Train model using `python main.py -cp conf -cn configs`

# Run the Web Demo version
- `streamlit run web.py`

# Train results

<p>
    <img src="assets\train_loss.jpg" alt="train_loss"/>
    <br>
    <em>Train loss of Deep Speech on 978 epochs</em>
</p>

<br>

<p>
    <img src="assets\validation_loss.jpg" alt="validation_loss"/>
    <br>
    <em>Validation loss of Deep Speech</em>
</p>

<br>

<p>
    <img src="assets\validation_wer.jpg" alt="validation_wer"/>
    <br>
    <em>Validation word error rate (mean wer) of Deep Speech</em>
</p>

## Note
- `sox` is audio backend for linux, `PySoundFile` is audio backend for windows

## Environment variable
- `HYDRA_FULL_ERROR=1`