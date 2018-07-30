# text-detection-ctpn-docker

## start

    docker run -p 8080:8080 -d cc861010/text-detect

## build

    docker build -t text-detect:first .

## test
	
    docker run --rm -it -v $PWD:/srv valian/docker-python-opencv-ffmpeg /bin/bash
