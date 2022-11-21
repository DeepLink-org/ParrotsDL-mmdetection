#!/bin/bash
set -x
set -o pipefail

# 0. check the most important SMART_ROOT
echo  "!!!!!SMART_ROOT is" $SMART_ROOT
if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

# 1. set env_path and build soft links for mm configs
if [[ $PWD =~ "mmdet" ]]
then 
    pyroot=$PWD
else
    pyroot=$PWD/mmdet
fi
echo $pyroot
if [ -d "$pyroot/algolib/configs" ]
then
    rm -rf $pyroot/algolib/configs
    ln -s $pyroot/configs $pyroot/algolib/
else
    ln -s $pyroot/configs $pyroot/algolib/
fi

# 2. build file folder for save log and set time
mkdir -p algolib_gen/mmdet/$3
now=$(date +"%Y%m%d_%H%M%S")

# 3. set env variables
export PYTORCH_VERSION=1.4
export MODEL_NAME=$3
export FRAME_NAME=mmdet    #customize for each frame
export PARROTS_DEFAULT_LOGGER=FALSE
export PYTHONPATH=$pyroot:$PYTHONPATH
export PYTHONPATH=${SMART_ROOT}:$PYTHONPATH

# 4. init
export PYTHONPATH=${SMART_ROOT}/common/sites:$PYTHONPATH

# 5. build necessary parameter
partition=$1  
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
EXTRA_ARGS=${array[@]:3}
EXTRA_ARGS=${EXTRA_ARGS//--resume/--resume-from}
SRUN_ARGS=${SRUN_ARGS:-""}

# 6. model list
case $MODEL_NAME in
    "mask_rcnn_r50_fpn_1x_coco")
        FULL_MODEL="mask_rcnn/mask_rcnn_r50_fpn_1x_coco"
        ;;
    "mask_rcnn_r101_fpn_1x_coco")
        FULL_MODEL="mask_rcnn/mask_rcnn_r101_fpn_1x_coco"
        ;;
    "retinanet_r50_fpn_1x_coco")
        FULL_MODEL="retinanet/retinanet_r50_fpn_1x_coco"
        ;;
    "ssd300_coco")
        FULL_MODEL="ssd/ssd300_coco"
        ;;
    "ssd300_voc0712")
        FULL_MODEL="pascal_voc/ssd300_voc0712"
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
    "grid_rcnn_r50_fpn_gn-head_2x_coco")
        FULL_MODEL="grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco"
        ;;
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
        FULL_MODEL="panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco"
        ;;
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
    "gfl_r50_fpn_1x_coco")
        FULL_MODEL="gfl/gfl_r50_fpn_1x_coco"
        ;;
    "autoassign_r50_fpn_8x2_1x_coco")
       FULL_MODEL="autoassign/autoassign_r50_fpn_8x2_1x_coco"
       ;;
    "faster_rcnn_r101_fpn_1x_coco")
       FULL_MODEL="faster_rcnn/faster_rcnn_r101_fpn_1x_coco"
       ;;
    "solo_r50_fpn_1x_coco")
       FULL_MODEL="solo/solo_r50_fpn_1x_coco"
       ;;
    "mask_rcnn_swin-t-p4-w7_fpn_1x_coco")
       FULL_MODEL="swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco"
       ;;
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

# 7. set port and choice model
port=`expr $RANDOM % 10000 + 20000`
file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

# 8. run model
srun -p $1 -n$2\
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py --config=$pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work_dir algolib_gen/${FRAME_NAME}/${MODEL_NAME} --options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now