#!/usr/bin/env bash
stage=$1

MODELS_PATH="/home/syl20/.local/share/tts"
RECIPE="recipes/vctk/fast_pitch"
MODEL_NAME="tts_models/en/vctk/fast_pitch"

# preprocess
if [ $stage -eq 0 ]; then
    DATA_DIR=recipes/vctk/VCTK-Corpus-0.92
    DATA_DIR_RESAMPLED=${DATA_DIR}-22k
    python TTS/bin/resample.py \
        --input_dir ${DATA_DIR} \
        --output_sr 22050 \
        --output_dir ${DATA_DIR_RESAMPLED} \
        --file_ext flac \
        --n_jobs 110
fi

# train
if [ $stage -eq 1 ]; then
    MODEL_PATH=${RECIPE}/fast_pitch_vctk-July-29-2022_09+24PM-c8a1f892
   
    python -m trainer.distribute \
        --gpus "0, 1, 2, 3, 4, 5, 6, 7" \
        --script ${RECIPE}/train_fast_pitch.py \
        --continue_path ${MODEL_PATH}
fi
if [ $stage -eq 2 ]; then
    
    # tts --model_name ${MODEL_NAME} --list_speaker_idxs  # list the possible speaker IDs.
    # tts --text "This is an example voice from V.C.T.K." --out_path output/speech.wav --model_name ${MODEL_NAME} --speaker_idx 'VCTK_p351'
    MODEL_PATH=${RECIPE}/fast_pitch_vctk-July-29-2022_09+24PM-c8a1f892
    # MODEL_PATH=${RECIPE}/fast_pitch_vctk_rishav-July-30-2022_02+54AM-c8a1f892
    # MODEL_PATH=${RECIPE}/fast_pitch_vctk-August-01-2022_05+11PM-c8a1f892
    # MODEL_PATH=/home/syl20/.local/share/tts/tts_models--en--vctk--fast_pitch 
    # model_file.pth best_model.pth
    tts --config_path ${MODEL_PATH}/config.json \
        --model_path ${MODEL_PATH}/best_model.pth \
        --text "This is an example voice from V.C.T.K. training" \
        --out_path output/output.wav \
        --speaker_idx 'VCTK_p351' \
        # --speaker_idx 'rishav'


        # --speaker_idx 'VCTK_p351' \
        # --vocoder_name vocoder_models/en/vctk/hifigan_v2
        # speaker embedder external
        # check-hifigan
        # VCTK
        # --speaker_idx 'rishav'
        # --vocoder_path path/to/vocoder.pth \
        # --vocoder_config_path path/to/vocoder_config.json \
fi


# finetune
if [ $stage -eq 3 ]; then
    RECIPE="recipes/vctk/fast_pitch"

    python -m trainer.distribute \
        --gpus "0, 1, 2, 3, 4, 5, 6, 7" \
        --script ${RECIPE}/train_fast_pitch_finetune.py \
        --restore_path ${MODELS_PATH}/tts_models--en--vctk--fast_pitch/model_file.pth

fi