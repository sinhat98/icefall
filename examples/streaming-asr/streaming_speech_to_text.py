#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-onnx Python API
# with endpoint detection.
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
# to download pre-trained models

import argparse
import sys
from pathlib import Path

import wave
import sherpa_onnx
import numpy as np
from threading import Thread


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default="audio.wav",
        help="Path to the audio file",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        required=True,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
        The file containing hotwords, one words/phrases per line, and for each
        phrase the bpe/cjkchar are separated by a space. For example:

        ▁HE LL O ▁WORLD
        你 好 世 界
        """,
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="""
        The hotword score of each token for biasing word/phrase. Used only if
        --hotwords-file is given.
        """,
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )

    return parser.parse_args()


def create_recognizer(args):
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,  # it essentially disables this rule
        decoding_method=args.decoding_method,
        provider=args.provider,
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
        blank_penalty=args.blank_penalty,
    )
    return recognizer


def audio_chunk_generator(audio_file: str, chunk_size: int):
    with wave.open(audio_file, "rb") as wf:
        assert wf.getnchannels() == 1, "Only mono audio is supported"
        assert wf.getsampwidth() == 2, "Only 16-bit audio is supported"
        assert wf.getframerate() == 16000, "Only 16 kHz audio is supported"
        assert wf.getcomptype() == "NONE", "Only PCM audio is supported"
        
        chunk_count = 0

        while True:
            chunk = wf.readframes(chunk_size)
            chunk_count += 1
            if not chunk:
                break
            # Read frames and convert from bytes to numpy array with dtype = int16
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            # Convert int16 to float32 and scale it to the range [-1.0, 1.0]
            audio_data = audio_data.astype(np.float32) / 32768.0
            yield chunk_count, audio_data

def main():
    args = get_args()

    recognizer = create_recognizer(args)
    # print("Started! Please speak")

    # The model is using 16 kHz, we use 48 kHz here to demonstrate that
    # sherpa-onnx will do resampling inside.
    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    

    stream = recognizer.create_stream()

    last_result = ""
    segment_id = 0
    import time
    print("Start decoding")
    # import pdb; pdb.set_trace()
    for chunk_count, samples in audio_chunk_generator(args.audio_file, samples_per_read):
        
        samples = samples.reshape(-1)
        
        # サンプル全体がゼロに近い値かをチェック
        # if np.allclose(samples, 0, atol=1e-6):
        #     print("No sound detected")
        # else:
        #     print("Sound detected")

        stream.accept_waveform(16000, samples)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        time.sleep(0.1)

        #is_endpoint = recognizer.is_endpoint(stream)

        result = recognizer.get_result(stream)
        # print(f"Decoding result: {result}")
        
        if result and (last_result != result):
            last_result = result
            # print("\r{}:{}".format(segment_id, result), end="", flush=True)
            print(chunk_count*0.1, result)
        # if is_endpoint:
        #     if result:
        #         print("\r{}:{}".format(segment_id, result), flush=True)
        #         segment_id += 1
        #     recognizer.reset(stream)


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")