export PATH=/mnt/cache/share/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/mnt/lustre/cache/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/mnt/cache/share/cuda-11.2

export PATH=$PATH:~/.local/bin
export PATH=/mnt/cache/share/gcc/gcc-5.4/bin/:$PATH


export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gmp-4.3.2/lib:/mnt/cache/share/gcc/mpfr-2.4.2/lib:/mnt/cache/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH

# export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gcc-5.4/lib64/:$LD_LIBRARY_PATH
# export PATH=/mnt/lustre/share/gcc/gcc-5.4/bin/:/mnt/cache/jizhenghao/.local/bin/:$PATH

export PATH=/mnt/cache/share/cmake3.8/bin/:$PATH

pip install -r requirements.txt
srun -p DBX16 --gres=gpu:1 --quotatype=spot --cpus-per-task=4 -N1 -n1 pip install -v -e .
