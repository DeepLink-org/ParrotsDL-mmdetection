#!/bin/bash
set -x

# 0. build soft link for mm configs
workdir=$(cd $(dirname $1); pwd)
if [[ "$workdir" =~ "submodules/mmdet" ]]
then 
    if [ -d "$workdir/algolib/configs" ]
    then
        rm -rf $workdir/algolib/configs
        ln -s $workdir/configs $workdir/algolib/
    else
        ln -s $workdir/configs $workdir/algolib/
    fi
else
    if [ -d "$workdir/submodules/mmdet/algolib/configs" ]
    then
        rm -rf $workdir/submodules/mmdet/algolib/configs
        ln -s $workdir/submodules/mmdet/configs $workdir/submodules/mmdet/algolib/
    else
        ln -s $workdir/submodules/mmdet/configs $workdir/submodules/mmdet/algolib/
    fi
fi

# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmdet/$3
export PYTORCH_VERSION=1.4
# 2. set time
now=$(date +"%Y%m%d_%H%M%S")

# 3. set env 
path=$PWD
if [[ "$path" =~ "submodules/mmdet" ]]
then 
    pyroot=$path
    comroot=$path/../..
    init_path=$path/..
else
    pyroot=$path/submodules/mmdet
    comroot=$path
    init_path=$path/submodules
fi
echo $pyroot
export PYTHONPATH=$comroot:$pyroot:$PYTHONPATH
export MODEL_NAME=$3
export FRAME_NAME=mmdet    #customize for each frame

# init_path
export PYTHONPATH=$init_path/common/sites/:$PYTHONPATH # necessary for init

# 4. build necessary parameter
partition=$1  
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
EXTRA_ARGS=${array[@]:3}
EXTRA_ARGS=${EXTRA_ARGS//--resume/--resume-from}
SRUN_ARGS=${SRUN_ARGS:-""}

# 5. model choice
export PARROTS_DEFAULT_LOGGER=FALSE

case $MODEL_NAME in
    "mask_rcnn_r50_fpn_1x_coco")
        FULL_MODEL="mask_rcnn/mask_rcnn_r50_fpn_1x_coco"
        ;;
    "retinanet_r50_fpn_1x_coco")
        FULL_MODEL="retinanet/retinanet_r50_fpn_1x_coco"
        ;;
    "ssd300_coco")
        FULL_MODEL="ssd/ssd300_coco"
        ;;
    "faster_rcnn_r50_fpn_1x_coco")
        FULL_MODEL="faster_rcnn/faster_rcnn_r50_fpn_1x_coco"
        ;;
    "retinanet_r50_fpn_fp16_1x_coco")
        FULL_MODEL="retinanet/retinanet_r50_fpn_fp16_1x_coco"
        ;;
    "cascade_mask_rcnn_r50_fpn_1x_coco")
        FULL_MODEL="cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco"
        ;;
    "yolov3_d53_320_273e_coco")
        FULL_MODEL="yolo/yolov3_d53_320_273e_coco"
        ;;
    "deformable_detr_r50_16x2_50e_coco")
        FULL_MODEL="deformable_detr/deformable_detr_r50_16x2_50e_coco"
        ;;
    # "grid_rcnn_r50_fpn_gn-head_1x_coco")
    #     FULL_MODEL="grid_rcnn/grid_rcnn_r50_fpn_gn-head_1x_coco"
    #     ;;
    # 注：grid_rcnn_r50_fpn_gn-head_1x_coco模型存在问题，详见 https://jira.sensetime.com/browse/PARROTSXQ-7589
    "point_rend_r50_caffe_fpn_mstrain_1x_coco")
        FULL_MODEL="point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco"
        ;;
    "detr_r50_8x2_150e_coco")
        FULL_MODEL="detr/detr_r50_8x2_150e_coco"
        ;;
    "centernet_resnet18_140e_coco")
        FULL_MODEL="centernet/centernet_resnet18_140e_coco"
        ;;
    "yolact_r50_8x8_coco")
        FULL_MODEL="yolact/yolact_r50_8x8_coco"
        ;;    
    "panoptic_fpn_r50_fpn_1x_coco")
<<<<<<< HEAD
        FULL_MODEL="panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco"
        ;;
=======
>>>>>>> 95f4f38925dbc3e9f6b021cabf2c11b5657aee7f
    # "htc_r50_fpn_1x_coco")
    #     FULL_MODEL="htc/htc_r50_fpn_1x_coco"
    #     ;;
    # htc模型有问题，详见https://jira.sensetime.com/browse/PARROTSXQ-7865?filter=-2
    "decoupled_solo_r50_fpn_1x_coco")
        FULL_MODEL="solo/decoupled_solo_r50_fpn_1x_coco"
        ;;
    "fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco")
        FULL_MODEL="fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco"
        ;;
    # "yolox_tiny_8x8_300e_coco")
    #    FULL_MODEL="yolox/yolox_tiny_8x8_300e_coco"
    #    ;;
    # yolox模型有问题，详见https://jira.sensetime.com/browse/PARROTSXQ-7812?filter=-2
    # "gfl_r50_fpn_1x_coco")
    #    FULL_MODEL="gfl/gfl_r50_fpn_1x_coco"
    #    ;;
    # gfl模型有问题，详见https://jira.sensetime.com/browse/PARROTSXQ-7810?filter=-2
    # "autoassign_r50_fpn_8x2_1x_coco")
    #    FULL_MODEL="autoassign/autoassign_r50_fpn_8x2_1x_coco"
    #    ;;
    # autoassign模型有问题，详见https://jira.sensetime.com/browse/PARROTSXQ-7809?filter=-2
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

port=`expr $RANDOM % 10000 + 20000`

set -x

file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work_dir algolib_gen/${FRAME_NAME}/${MODEL_NAME} --options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
