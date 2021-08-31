# building image

```
cd ~/src/deeplexicon/dockerfiles/cpu
name=deeplexicon; version=1.2.0
echo $name:$version

docker build --pull -t lpryszcz/$name:$version .
docker tag lpryszcz/$name:$version lpryszcz/$name:latest

docker push lpryszcz/$name:$version && docker push lpryszcz/$name:latest
```

# testing

```bash
source ~/src/venv/deeplexicon-gpu/bin/activate
cd ~/test/deeplexicon/test
pip install ont_fast5_api h5py==2.10 Keras==2.2.4 Pandas PyTs==0.8.0 Scikit-learn numba==0.53 TensorFlow-gpu==1.13.1 

time ~/src/deeplexicon/deeplexicon_sub.py dmux -g -p $d -m ~/src/deeplexicon/models/resnet20-final.h5 > $d.demux-gpu.tsv

d=RNA050821.vbz
time docker run -u $UID:$GID -v `pwd`:/data lpryszcz/deeplexicon:1.2.0 deeplexicon_sub.py dmux -p /data/$d -m deeplexicon/models/resnet20-final.h5 > $d.demux.docker.tsv

# gpu
time docker run --gpus all -u $UID:$GID -v `pwd`:/data lpryszcz/deeplexicon:1.2.0-gpu deeplexicon_sub.py dmux -p /data/$d -m deeplexicon/models/resnet20-final.h5 > $d.demux.docker-gpu.tsv
```
