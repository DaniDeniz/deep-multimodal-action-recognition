# Multimodal Action Recognition
Deep multimodal action recognition for Habit Tracking from Video and Heart Rate (pulses) information

This repository is from the work: *Deep multimodal habit tracking system: A user-adaptive approach for low-power
embedded systems* published at *Journal of Signal Processing Systems*.

Here, we provide the Deep Learning models architectures using the Tensorflow keras framework and the weights of the
trained models.

## Pre-requisites
Firstly, create a python virtualenv to run these models. Then, enter the python virtualenv.
```bash
python3 -m virtualenv -p python3 venv
source venv/bin/activate
```

Clone this repository to your computer
```bash
git clone https://github.com/DaniDeniz/deep-multimodal-action-recognition.git
cd deep-multimodal-action-recognition
```

Install the python required modules.
```bash
pip install -r requirements.txt
```

## Demo Tutorial
Refer to [demo_tutorial.ipynb](demo_tutorial.ipynb) to see an example of how to load the designed models with their weights
and how to do inferences to perform action recognition using video and heart rate information.

The tutorial also shows the recognition performance of the architectures when identifying risky situations.

## Citation
Bibtext

##License
[BSD 3-Clause License](LICENSE)
