#!/bin/bash

# help
function usage()
{
    cat <<EOM
Usage: bash $0 [options]
Options:
  --master_port PORT               Master port (default=12345)
  --gpu GPU                        GPU index (default=0)
  -f, --config_path PATH           Path of config file (required)
  -o, --override_config_path PATH  Path of override config file (optional)
  --output_dir PATH                Output directory (optional)
  --exp_name NAME                  Experiment name (optional)
  --model_path PATH                Path of checkpoint to test (optional)
  -h, --help                       Print help
EOM

    exit 1
}


# parser
function set_options()
{
    arguments=$(getopt --options f:o:h \
                       --longoptions master_port:,gpu:,config_path:,override_config_path:,output_dir:,exp_name:,model_path:,help \
                       --name $(basename $0) \
                       -- "$@")

    eval set -- "$arguments"

    while true
    do
        case "$1" in
            --master_port)
                MASTER_PORT=$2
                shift 2
                ;;
            --gpu)
                GPU=$2
                shift 2
                ;;
            -f | --config_path)
                CONFIG_PATH=$2
                shift 2
                ;;
            -o | --override_config_path)
                OVERRIDE_CONFIG_PATH=$2
                shift 2
                ;;
            --output_dir)
                OUTPUT_DIR=$2
                shift 2
                ;;
            --exp_name)
                EXP_NAME=$2
                shift 2
                ;;
            --model_path)
                MODEL_PATH=$2
                shift 2
                ;;
            --)
                shift
                break
                ;;
            -h | --help)
                usage
                ;;
        esac
    done
}

# default
MASTER_PORT=12345
GPU="0"

# parsing
set_options "$@"

# check required arguments
if [ -z "$CONFIG_PATH" ]; then
    echo "Error: config_path is required"
    usage
fi

# print parsed arguments
echo "Arguments"
echo -e "\tMASTER_PORT: ${MASTER_PORT}"
echo -e "\tGPU: ${GPU}"
echo -e "\tCONFIG_PATH: ${CONFIG_PATH}"
echo -e "\tOVERRIDE_CONFIG_PATH: ${OVERRIDE_CONFIG_PATH}"
echo -e "\tOUTPUT_DIR: ${OUTPUT_DIR}"
echo -e "\tEXP_NAME: ${EXP_NAME}"
echo -e "\tMODEL_PATH: ${MODEL_PATH}"

# set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU

# run downstream training
MAIN_ARGS="--config_path $CONFIG_PATH"

if [ ! -z ${OVERRIDE_CONFIG_PATH} ]; then
    MAIN_ARGS="$MAIN_ARGS \
               --override_config_path $OVERRIDE_CONFIG_PATH"
fi
if [ ! -z ${OUTPUT_DIR} ]; then
    MAIN_ARGS="$MAIN_ARGS \
               --output_dir $OUTPUT_DIR"
fi
if [ ! -z ${EXP_NAME} ]; then
    MAIN_ARGS="$MAIN_ARGS \
               --exp_name $EXP_NAME"
fi
if [ ! -z ${MODEL_PATH} ]; then
    MAIN_ARGS="$MAIN_ARGS \
               --model_path $MODEL_PATH"
fi

echo "Run segmentation model testing..."
cd src
python test.py $MAIN_ARGS
