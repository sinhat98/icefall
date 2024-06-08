expdir=$1
from_micrpohone=$2

token_file=${expdir}/tokens.txt
encoder_file=${expdir}/encoder-epoch-30-avg-3-with-averaged-model.onnx
decoder_file=${expdir}/decoder-epoch-30-avg-3-with-averaged-model.onnx
joiner_file=${expdir}/joiner-epoch-30-avg-3-with-averaged-model.onnx


if [ ${from_micrpohone} == "true" ]; then
  python speech-recognition-from-microphone-with-endpoint-detection.py \
    --tokens=${token_file} \
    --encoder=${encoder_file} \
    --decoder=${decoder_file} \
    --joiner=${joiner_file}
else
  python streaming_speech_to_text.py \
    --audio_file audio.wav \
    --tokens=${token_file} \
    --encoder=${encoder_file} \
    --decoder=${decoder_file} \
    --joiner=${joiner_file}
fi