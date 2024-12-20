# GenDARA 2025 Challenge - Generative Data Augmentation for Room Acoustics for Speaker Distance Estimation

## Introduction

Blah

## Challenge Details and Rules

Blah

## Baseline System

We retrain a SoTA Speaker Distance Estimation Model on the C4DM room impulse response dataset and the VCTK speech dataset. 
This is the baseline SDE system, and the checkpoint, training script, and model architecture are provided in `./GenDARA/sde_model`. 

We set aside speaker id XXX, YYY, ZZZ, WWW as test speech.

When finetuning this SDE model, please do not use test speakers.

## Submission Instructions

We use ICASSP 2025’s submission system on CMT.

- Login as an “Author” at <https://cmt3.research.microsoft.com/ICASSP2025/>
- Choose “+Create new submission…” menu on the top left
- Choose the workshop “Satellite Workshop: Generative Data Augmentation for Real-World Signal Processing Applications”
- Fill out the author form and choose “Challenge: Room Acoustics and Speaker Distance Estimation” as the primary subject area
- Once after the submission of the two-page summary document, you will be able to see in the author console that your submission is created. On the rightmost column of your submission, you can upload the “supplementary material” which must contain all the zipped submission files.
- As described in the track details, participants are expected to submit 102 wav files for track 1 and single .csv file with 480 distance estimates for track 2.
- Please follow the directory format as below. Thank you.
