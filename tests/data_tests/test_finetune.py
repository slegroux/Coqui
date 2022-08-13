import functools
import random
import unittest
import torch
import os

from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
from TTS.tts.configs.fast_pitch_config import FastPitchConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.speakers import SpeakerManager
from trainer import Trainer, TrainerArgs

# Fixing random state to avoid random fails
torch.manual_seed(0)
output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    name="json_manifest",
    meta_file_train="rishav_5min.train.json",
    path="/home/syl20/data/en/webex_speakers/en",
    language="en",
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=23.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = FastPitchConfig(
    run_name="fast_pitch_vctk_rishav",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    compute_input_seq_cache=True,
    precompute_num_workers=4,
    compute_f0=True,
    f0_cache_path=os.path.join(output_path, "f0_cache"),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    min_text_len=0,
    max_text_len=500,
    min_audio_len=0,
    max_audio_len=500000,
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=True,
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = ForwardTTS(config, ap, tokenizer)
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# trainer.fit()