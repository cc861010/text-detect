# text-detection-ctpn-docker (base from https://github.com/eragonruan/text-detection-ctpn.git)

## start

    docker run -p 8080:8080 -d cc861010/text-detect

## build

    docker build -t text-detect:first .

## test
	
    docker run --rm -it -v $PWD:/srv valian/docker-python-opencv-ffmpeg /bin/bash
