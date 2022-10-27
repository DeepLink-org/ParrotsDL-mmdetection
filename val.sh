port=`expr $RANDOM % 10000 + 20000`
# srun -p camb_mlu290 -n8 --gres=mlu:8 --ntasks-per-node 8 --job-name=SoloV python tools/test.py configs/solo/solo_r50_fpn_1x_coco.py ./solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth --out=outputs.pkl --eval bbox segm --launcher=slurm --cfg-options dist_params.port=$port

srun -p camb_mlu290 -n1 --gres=mlu:1 --ntasks-per-node 1 --job-name=SoloV python tools/test.py configs/solo/solo_r50_fpn_1x_coco.py ./solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth --out=outputs.pkl --eval bbox segm --launcher=slurm --cfg-options dist_params.port=$port
