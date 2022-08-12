# Train and evaluate CNN model

python showcase/scripts/train_tensorflow.py --config cnn-base-model-1
python showcase/scripts/train_tensorflow.py --config cnn-base-model-2
python showcase/scripts/train_tensorflow.py --config cnn-curve-pretrained
python showcase/scripts/train_tensorflow.py --config cnn-curve-resume
python showcase/scripts/evaluate_tensorflow.py --config cnn-curve-evaluate-trained
python showcase/scripts/evaluate_tensorflow.py --config cnn-curve-evaluate-point

# Train and evaluate CNNBN model

python showcase/scripts/train_tensorflow.py --config cnnbn-base-model-1
python showcase/scripts/train_tensorflow.py --config cnnbn-base-model-2
python showcase/scripts/train_tensorflow.py --config cnnbn-curve-pretrained
python showcase/scripts/train_tensorflow.py --config cnnbn-curve-resume
python showcase/scripts/evaluate_tensorflow.py --config cnnbn-curve-evaluate-trained
python showcase/scripts/evaluate_tensorflow.py --config cnnbn-curve-evaluate-point