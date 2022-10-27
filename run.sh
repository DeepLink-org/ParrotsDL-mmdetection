
port=`expr $RANDOM % 10000 + 20000`
time=$(date "+%Y%m%d%H%M%S")
# nohup srun -p camb_mlu290 -n8 --gres=mlu:8 --ntasks-per-node 8 --job-name=Centernet python tools/train.py --config=configs/centernet/centernet_resnet18_dcnv2_140e_coco.py --launcher=slurm --resume-from=work_dirs/centernet_resnet18_dcnv2_140e_coco/latest.pth --cfg-options dist_params.port=$port >log/train_centernet_${time}.log 2>&1 &
# srun -p camb_mlu290 -n8 --gres=mlu:8 --ntasks-per-node 8 --job-name=Centernet python tools/train.py --config=configs/centernet/centernet_resnet18_dcnv2_140e_coco.py --launcher=slurm --cfg-options dist_params.port=$port

# nohup srun -p camb_mlu290 -n8 --gres=mlu:8 --ntasks-per-node 8 --job-name=Centernet python tools/train.py --config=configs/centernet/centernet_resnet18_dcnv2_140e_coco.py --launcher=slurm --resume-from=work_dirs/centernet_resnet18_dcnv2_140e_coco/latest.pth --cfg-options dist_params.port=$port >log/train_centernet_${time}.log 2>&1 &
# nohup srun -p camb_mlu290 -n8 --gres=mlu:8 --ntasks-per-node 8 --job-name=Centernet python tools/train.py --config=configs/centernet/centernet_resnet18_dcnv2_140e_coco.py --launcher=slurm --cfg-options dist_params.port=$port >log/train_centernet_${time}.log 2>&1 &

srun -p camb_mlu290 -n1 --gres=mlu:1 --ntasks-per-node 1 --job-name=Centernet python tools/train.py --config=configs/centernet/centernet_resnet18_140e_coco.py --launcher=slurm --cfg-options dist_params.port=$port
