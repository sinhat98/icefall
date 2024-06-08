# ストリーミング音声認識

# 構成内容

## pytorchファイル

- epoch-28.pt
- epoch-29.pt
- epoch-30.pt

## ONNXファイル

- encoder-epoch-30-avg-3-with-averaged-model.onnx
- decoder-epoch-30-avg-3-with-averaged-model.onnx
- joiner-epoch-30-avg-3-with-averaged-model.onnx

## 言語モデル

- token.txt
- bpe.model

# 実行方法

## pytorchファイル

[icefallライブラリのstreaming_decode.py](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/decode.py)を用いて実行

```
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 30 \
    --avg 3 \
    --exp-dir ./{pytorchファイルが存在するディレクトリ} \
    --bpe-model ./bpe.model \
    --max-duration 200 \
    --decode-chunk-len 64 \
    --num-left-chunks 3 \
    --decoding-method "greedy_search" \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64
```

## onnxファイル

[sherpa-onnxライブラリのspeech-recognition-from-microphone-with-endpoint-detection.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py)を用いて実行

```
python3 ./python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py \
  --tokens=./tokens.txt \
  --encoder=./encoder-epoch-30-avg-3-with-averaged-model.onnx \
  --decoder=./decoder-epoch-30-avg-3-with-averaged-model.onnx \
  --joiner=./joiner-epoch-30-avg-3-with-averaged-model.onnx
```