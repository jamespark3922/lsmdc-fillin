#!/bin/bash
mkdir -p data 

if [ ! -d "data/i3d" ]
then
	echo "Downloading i3d features..."
	wget http://lsmdc19.berkeleyvision.org/i3d_200.zip -nv -c --show-progress
	echo "Extracting i3d features in data/"
	unzip -q i3d_200.zip -d data
	mv data/i3d2 data/i3d
	rm i3d_200.zip
	echo "Saved i3d features in data/i3d/"
else 
	echo "Already extracted i3d features in data/i3d/ ! Remove the directory if you would like to try again"
fi
echo

if [ ! -d "data/fillin_data" ]
then
	echo "Downloading preprocessed data and face cluster features..."
	wget https://storage.googleapis.com/ai2-jamesp-public/lsmdc/fillin_data.zip -nv -c --show-progress
	echo "Extracting data... in data/"
	unzip fillin_data.zip -d data
	echo "Saved at data/fillin_data/"
	rm fillin_data.zip
else
	echo "Already extracted preprocessed data in data/fillin_data/ ! Please remove the directory data/fillin_data/ if you would like to try again."
fi

echo
echo "Done"
