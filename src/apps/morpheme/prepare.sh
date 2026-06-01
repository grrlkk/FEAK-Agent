#pip install -r requirements.txt

mkdir -p ./.data/multi30k/raw
gzip -d ./.data/multi30k/raw/train.en.gz
gzip -d ./.data/multi30k/raw/train.de.gz
gzip -d ./.data/multi30k/raw/val.en.gz
gzip -d ./.data/multi30k/raw/val.de.gz
gzip -d ./.data/multi30k/raw/test_2016_flickr.en.gz
gzip -d ./.data/multi30k/raw/test_2016_flickr.de.gz
