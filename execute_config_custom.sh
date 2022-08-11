# Train and evaluate CNN model

python showcase/scripts/train_classification.py --config cnn-base-model-1
python showcase/scripts/train_classification.py --config cnn-base-model-2
python showcase/scripts/train_classification.py --config cnn-curve-pretrained
python showcase/scripts/train_classification.py --config cnn-curve-resume
python showcase/scripts/evaluate_classification.py --config cnn-curve-evaluate-init
python showcase/scripts/evaluate_classification.py --config cnn-curve-evaluate-trained
python showcase/scripts/evaluate_classification.py --config cnn-curve-evaluate-point

# Train and evaluate CNNBN model

python showcase/scripts/train_classification.py --config cnnbn-base-model-1
python showcase/scripts/train_classification.py --config cnnbn-base-model-2
python showcase/scripts/train_classification.py --config cnnbn-curve-pretrained
python showcase/scripts/train_classification.py --config cnnbn-curve-resume
python showcase/scripts/evaluate_classification.py --config cnnbn-curve-evaluate-init
python showcase/scripts/evaluate_classification.py --config cnnbn-curve-evaluate-trained
python showcase/scripts/evaluate_classification.py --config cnnbn-curve-evaluate-point

# Train and evaluate MLP model

python showcase/scripts/train_regression.py --config mlp-base-model-1
python showcase/scripts/train_regression.py --config mlp-base-model-2
python showcase/scripts/train_regression.py --config mlp-curve-pretrained
python showcase/scripts/train_regression.py --config mlp-curve-resume
python showcase/scripts/evaluate_regression.py --config mlp-curve-evaluate-init
python showcase/scripts/evaluate_regression.py --config mlp-curve-evaluate-trained
python showcase/scripts/evaluate_regression.py --config mlp-curve-evaluate-point
