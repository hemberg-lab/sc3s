ARG ubuntu_version=18.04

FROM ubuntu:$ubuntu_version

RUN set -xe \
    && apt-get -y update \
    && apt-get -y install git \
    && apt-get -y install python3.6 \
    && apt-get -y install python3-dev \
    && apt-get -y install python3-pip 

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade --no-cache \
	"numpy>=1.19.2,<2" \
	"pandas>=1.1.5" \
	"scikit-learn>=0.23.2" \
	"scanpy>=1.7.0" \
	"six>=1.15.0" \
	"setuptools>=52.0.0" \
	"joblib>=1.0.0" \
	"h5py>=3.1.0" \
	"seaborn" \
	"scipy" \
	"python-dateutil>=2.8.1"
