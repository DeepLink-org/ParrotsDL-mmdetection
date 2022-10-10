port=`expr $RANDOM % 10000 + 20000`
time=$(date "+%Y%m%d%H%M%S")
# nohup srun -p camb_mlu290 -n8 --gres=mlu:8 --ntasks-per-node 8 --job-name=Autoassign python tools/train.py --resume-from=work_dirs/autoassign_r50_fpn_8x2_1x_coco/latest.pth --config=configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py --launcher=slurm --cfg-options dist_params.port=$port >log/train_autoassign_${time}.log 2>&1 &
# srun -p camb_mlu290 -n1 --gres=mlu:1 --ntasks-per-node 1 --job-name=YoLov3 python tools/train.py  --config=configs/yolo/yolov3_d53_320_273e_coco.py --launcher=slurm --cfg-options dist_params.port=$port

srun -p camb_mlu290 -n8 --gres=mlu:8 --ntasks-per-node 8 --job-name=YoLov3 python tools/train.py  --config=configs/yolo/yolov3_d53_320_273e_coco.py --launcher=slurm --cfg-options dist_params.port=$port
