#! /bin/bash

#1#
# parsing the arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Two arguments are required."
    echo "Usage: bash docker_run.sh <container_name> <image_name>"
    exit 1

else
    CONTAINER_NAME=$1
    IMAGE_NAME=$2
fi


#2#
# Set optional arguments and flags

# check for V4L2 devices
V4L2_DEVICES=""

for i in {0..9}
do
	if [ -a "/dev/video$i" ]; then
		V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
	fi
done

# check for I2C devices
I2C_DEVICES=""

for i in {0..9}
do
	if [ -a "/dev/i2c-$i" ]; then
		I2C_DEVICES="$I2C_DEVICES --device /dev/i2c-$i "
	fi
done

# check for display
DISPLAY_DEVICE=""

if [ -n "$DISPLAY" ]; then
	# give docker root user X11 permissions
	xhost +si:localuser:root || sudo xhost +si:localuser:root

	DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH"
fi

# check for jtop
JTOP_SOCKET=""
JTOP_SOCKET_FILE="/run/jtop.sock"

if [ -S "$JTOP_SOCKET_FILE" ]; then
	JTOP_SOCKET="-v /run/jtop.sock:/run/jtop.sock"
fi

# extra flags
EXTRA_FLAGS=""

if [ -n "$HUGGINGFACE_TOKEN" ]; then
	EXTRA_FLAGS="$EXTRA_FLAGS --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN"
fi

# additional permission optional run arguments
OPTIONAL_PERMISSION_ARGS=""

if [ "$USE_OPTIONAL_PERMISSION_ARGS" = "true" ]; then
	OPTIONAL_PERMISSION_ARGS="-v /lib/modules:/lib/modules --device /dev/fuse --cap-add SYS_ADMIN --security-opt apparmor=unconfined"
fi

# check if sudo is needed
if [ $(id -u) -eq 0 ] || id -nG "$USER" | grep -qw "docker"; then
	SUDO=""
else
	SUDO="sudo"
fi

ROOT="$(dirname "$(readlink -f "$0")")"

#3#
# Run the docker container
ARCH=$(uname -i)

if [ $ARCH = "aarch64" ]; then

    set -x

    # If the container exists, remove it to avoid conflict
    if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
        docker rm $CONTAINER_NAME
    fi

    docker run -it --network=host --runtime=nvidia --shm-size=8g --gpus all \
        --volume /tmp/argus_socket:/tmp/argus_socket \
		--volume /etc/enctune.conf:/etc/enctune.conf \
		--volume /etc/nv_tegra_release:/etc/nv_tegra_release \
		--volume /var/run/dbus:/var/run/dbus \
		--volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
		--volume /var/run/docker.sock:/var/run/docker.sock \
        --volume /dev:/dev \
        $OPTIONAL_PERMISSION_ARGS $DATA_VOLUME $DISPLAY_DEVICE $V4L2_DEVICES $I2C_DEVICES $JTOP_SOCKET $EXTRA_FLAGS \
        --device /dev/ttyACM0 \
        --device /dev/input/js0 \
        --privileged\
        --name $CONTAINER_NAME \
        $IMAGE_NAME \
        /bin/bash
    
else
    exit 1
fi
