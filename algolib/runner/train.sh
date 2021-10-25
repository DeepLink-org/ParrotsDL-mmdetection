set -x
#!/bin/bash
# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmdet/$3
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
CONDA_ROOT=/mnt/cache/share/platform/env/miniconda3.6
MMCV_PATH=${CONDA_ROOT}/envs/${CONDA_DEFAULT_ENV}/mmcvs
mmcv_version=1.3.10
export PYTHONPATH=${MMCV_PATH}/${mmcv_version}:$PYTHONPATH

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
    "mask_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/mask_rcnn/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "retinanet_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/retinanet/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "ssd300_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/ssd/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "faster_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/faster_rcnn/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "retinanet_r50_fpn_fp16_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/fp16/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "fast_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/fast_rcnn/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "cascade_mask_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/cascade_rcnn/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "yolov3_d53_320_273e_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/yolo/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "fast_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/fast_rcnn/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "deformable_detr_r50_16x2_50e_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/deformable_detr/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "grid_rcnn_r50_fpn_gn-head_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/grid_rcnn/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "point_rend_r50_caffe_fpn_mstrain_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/point_rend/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "detr_r50_8x2_150e_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/detr/${MODEL_NAME}.py --launcher=slurm \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "yolact_r50_1x8_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/yolact/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    "centernet_resnet18_dcnv2_140e_coco")
set -x

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/centernet/${MODEL_NAME}.py --launcher=slurm  \
    --work_dir algolib_gen/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/mmdet/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
    ;;
    *)
      echo "invalid $MODEL_NAME"
      exit 1
      ;;
esac
