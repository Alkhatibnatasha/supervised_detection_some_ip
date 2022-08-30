# Supervised Intrusion Detection for SOME/IP protocol
SOME/IP is an automotive/embedded communication protocol which supports remote procedure calls, event notifications and the underlying serialization/wire format. In this project, we use deep learning algorithms to detect cyberattacks on SOME/IP. This repository contains the source code and data for our research paper: "Deep learning-based intrusion detection for SOME/IP protocol" by Natasha Alkhatib, Maria Mushtaq, Hadi Ghauch, Jean-Luc Danger.

## Dependencies ##
The project was developed with the following library versions. Running with other versions may crash or produce incorrect results.

* Python 3.8.10
* CUDA Version: 11.2
* torch==1.10.0
* numpy==1.19.5
* pandas==1.2.5
* scikit-learn==0.24.2
* json==2.0.9
* pickle==4.0

## Setup Instructions ## 
1. Clone this repo: git clone https://github.com/Alkhatibnatasha/supervised_detection_some_ip
2. Download SOME/IP dataset from this link and extract into output/config1/data/

## Experiment ## 

 `cd config1

 #Train the lstm algorithm
 python someip_lstm.py train
 #Test the lstm algorithm
 python someip_lstm.py test
 #Check the inference time of the lstm algorithm
 python someip_lstm.py time
 
 `
