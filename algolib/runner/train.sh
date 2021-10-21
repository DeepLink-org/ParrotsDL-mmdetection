#!/bin/bash
# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmdet
export PYTORCH_VERSION=1.4
# 2. set time
now=$(date +"%Y%m%d_%H%M%S")

# 3. set env 
path=$PWD
if [[ "$path" =~ "algolib/mmdet" ]]
then 
    pyroot=$path
    comroot=$path/../..
else
    pyroot=$path/algolib/mmdet
    comroot=$path
fi
echo $pyroot
export PYTHONPATH=$comroot:$pyroot:$PYTHONPATH
export MODEL_NAME=$3
export FRAME_NAME=mmdet

# 4. build necessary parameter
partition=$1  
name=$3
MODEL_NAME=$3
EXTRA_ARGS=${@:4}
g=$(($2<8?$2:8))

# 5. build optional parameter
SRUN_ARGS=${SRUN_ARGS:-""}

# 6. save log
# 避免mm系列重复打印
export PARROTS_DEFAULT_LOGGER=FALSE

case $MODEL_NAME in
    "mask_rcnn_r50_caffe_fpn_mstrain_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/mask_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "mask_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/mask_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "cascade_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/cascade_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "retinanet_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/retinanet/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "mask_rcnn_r50_fpn_fp16_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/fp16/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "ssd300_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/ssd/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "faster_rcnn_r50_fpn_fp16_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/fp16/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "mask_rcnn_x101_32x4d_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/mask_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "faster_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/faster_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "retinanet_r50_fpn_fp16_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/fp16/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "mask_rcnn_r101_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/mask_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "mask_rcnn_x101_64x4d_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/mask_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "fast_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/fast_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "cascade_mask_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/cascade_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAMEe}.log.$now
    ;;
    "mask_rcnn_x101_64x4d_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/mask_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/dcn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "fsaf_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/fsaf/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "yolov3_d53_320_273e_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/yolo/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "fast_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/fast_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "faster_rcnn_hrnetv2p_w18_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/hrnet/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "deformable_detr_r50_16x2_50e_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/deformable_detr/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    "grid_rcnn_r50_fpn_gn-head_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/grid_rcnn/${MODEL_NAME}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/train.${MODEL_NAME}.log.$now
    ;;
    *)
      echo "invalid $MODEL_NAME"
      exit 1
      ;;
esac
