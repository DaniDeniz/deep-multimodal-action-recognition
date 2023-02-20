# Multimodal Action Recognition
Deep multimodal action recognition for Habit Tracking from Video and Heart Rate (pulses) information

This repository is from the work: *Deep multimodal habit tracking system: A user-adaptive approach for low-power
embedded systems* published at *Journal of Signal Processing Systems* [1]().

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
[1] D. Deniz, G. Jimenez-Perera, R. Nolasco, J. Corral, F. Barranco. "Deep Multimodal Habit Tracking System: A User-adaptive Approach for Low-power Embedded Systems" in *Journal of Signal Processing Systems* (2023). doi: [10.1007/s11265-023-01840-4](https://doi.org/10.1007/s11265-023-01840-4)
```
@article{deniz2023deep,
	author={Deniz, Daniel
	and Jimenez-Perera, Gabriel
	and Nolasco, Ricardo
	and Corral, Javier
	and Barranco, Francisco},
	title={Deep Multimodal Habit Tracking System: A User-adaptive Approach for Low-power Embedded Systems},
	journal={Journal of Signal Processing Systems},
	year={2023},
	issn={1939-8115},
	doi={10.1007/s11265-023-01840-4}
}
```

## License
[BSD 3-Clause License](LICENSE)
