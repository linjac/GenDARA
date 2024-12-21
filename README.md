# GenDARA 2025 Challenge - Generative Data Augmentation for Room Acoustics for Speaker Distance Estimation

## Introduction

Blah

## Challenge Details and Rules

### Task 1: Augmenting RIR Data with Participant System

In Task 1, we aim to evaluate the participant's RIR generation system based on the RIRs it generates, although the main point of the challenge is to investigate its usefulness for the downstream task. At any rate, the goal of the RIR generation system is to generate RIRs from new source and receiver locations from a given set of RIRs collected within a room. The goal to augment sparsely collected RIR data, whose quality tend to be limited, to help improve the performance of downstream tasks such as speaker distance estimation, dereverberation, etc. To evaluate the RIR generation system performance, participants are to generate the RIRs at specified source-receiver locations in 20 rooms.

However, the point is that participants are allowed to build an RIR augmented dataset that best improves performance of a downstream task in Task 2. This may involve generating more RIRs beyond the requested source-receiver positions from the enrollment rooms.

#### Enrollment Data

We provide data on 20 different rooms in `enrollment_data`. Rooms 1-10 are simulated using Treble Technologies' wave-based simulator. Rooms 11-20 are a sampled from the GWA Dataset, which are simulated from a hybrid wave-based and geometrical acoustics simulator.

For Rooms 1-10, for each room we provide:

- 5 single-channel RIRs
- 5 8th-order HOA RIRs
- labeled source + receiver positions in `meta.csv`
- 3D models

For Rooms 11-20, for each room we provide:

- 5 single-channel RIRs
- labeled source + receiver positions in `meta.csv`

Additionally, we provide a control set of RIRs in Room_0 `enrollment_data/Room_0_data` for participants to calbirate their systems as necessary. Room_0 is a physical room at Treble's offices with variable wall absorption and furniture layout. 20 single-channel RIRs were measured in Room_0, and their simulated counterparts (same source and receiver positions) are provided. Also, a grid of virtual receivers were simulated and those simulated RIRs are provided:

- 20 measured single-channel RIRs `enrollment_data/Room_0_data/measured_rirs`
- 20 simulated single-channel & 8th-order HOA RIRs at measurement positions `enrollment_data/Room_0_data/simulated_rirs/measurement_positions`
- 405 simulated single-channel & 8th-order HOA RIRs at grid positions `enrollment_data/Room_0_data/simulated_rirs/measurement_positions`
- labeled source + receiver positions
- 3D model

#### Evaluation 1

As mentioned above, participant's RIR generation system is evaluated by the quality of the generated RIRs. The room, source, and receiver positions of the requested submission RIRs are found in `submission_folder/eval_1_rirs/meta_test_rir.csv`. The RIRs will be evaluated on their T60, DRR, and EDF similarity to the witheld reference RIRs. The RIR evaluation we will perform is shown in [this jupyter notebook](https://github.com/linjac/GenDARA/blob/main/evaluation_rir_room0.ipynb) on Room_0 as an example.

### Task 2: Improving Speaker Distance Estimation with Augmented RIR Data

In Task 2, participants are expected to improve a speaker distance estimation (SDE) model using the augmented RIR dataset generated from the sparse enrollment data in Task 1.

#### Evaluation 2

The participant's fine-tuned SDE systems must estimate the speaker distance from a test set of 480 reverberant speech audio in `submission_folder/eval_2_speaker_distance_estimates/test_audio`. The provided Baseline SDE system's estimates for the test audio are in [this .csv file](https://github.com/linjac/GenDARA/blob/main/submission_folder/speaker_distance_estimates/test_speaker_distance_estimates.csv). Participants are asked to submit a .csv file in the same format containing their updated distance estimates in meters.

The submitted distance estimates will be evaluated on the absolute distance error and the percentage distance error.

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

```
{Team name}_submission_folder.zip
├── eval_1_rirs
│   ├── Room_1_Source_1_Receiver_0_IR_SC.wav
│   ├── Room_1_Source_3_Receiver_3_IR_SC.wav
|   ├── Room_1_Source_2_Receiver_5_IR_SC.wav
│   ...
│   └── Room_20_Source_1_Receiver_10_GWA_IR_SC.wav
|
└── eval_2_speaker_distance_estimation
    └── test_speaker_distance_estimates.csv
```