source /home/pkuhetu/ls/bashrc
conda activate hetu-py

HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
source ${HETU_HOME}/hetu.exp
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

export HETU_MEMORY_PROFILE=WARN
export HETU_MEMORY_LOG_FILE=""
export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=INFO
if [ -z $HETU_EVENT_TIMING ]; then
    export HETU_EVENT_TIMING=OFF
fi

export NCCL_DEBUG=VERSION
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

if [ -z $DP_BUCKET ]; then
    export DP_BUCKET=ON
fi

# if [ $DP_BUCKET = "ON" ] || [ $BUCKET_NUM = 16 ]; then
if [ $DP_BUCKET = "ON" ]; then
    export HETU_MAX_SPLIT_SIZE_MB=10240
    export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=0
else 
    export HETU_MAX_SPLIT_SIZE_MB=200
    export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20
fi

# experiment env
if [ -z $EXPR_DATA_DISPATCH ]; then
    export EXPR_DATA_DISPATCH=BALANCE
fi

if [ -z $EXPR_SENSITIVITY ]; then
    export EXPR_SENSITIVITY=OFF
fi

if [ -z $EXPR_EFFECTIVENESS ]; then
    export EXPR_EFFECTIVENESS=OFF
fi

if [ -z $EXPR_CUSTOM_DISTRIBUTION ]; then
    export EXPR_CUSTOM_DISTRIBUTION=OFF
fi

if [ -z $EXPR_SIMULATE ]; then
    export EXPR_SIMULATE=OFF
fi

if [ -z ${EXPR_SCHEME_PROPOSAL} ]; then
    export EXPR_SCHEME_PROPOSAL=ON
fi

if [ -z ${EXPR_DEPLOY_PATTERN} ]; then
    export EXPR_DEPLOY_PATTERN=PRUNE
fi

if [ -z ${BUCKET_PLAN} ]; then
    export BUCKET_PLAN=DYNAMIC
fi

echo "env done"