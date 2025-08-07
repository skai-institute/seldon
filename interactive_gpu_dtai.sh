#srun --account=bejy-dtai-gh --partition=ghx4-interactive --nodes=1 --tasks-per-node=72 --exclusive --gres=gpu:h100:4 --mem=0 --job-name=gpu-interactive --time 02:00:00 --pty bash -i
srun --account=bejy-dtai-gh --partition=ghx4-interactive --nodes=1 --tasks-per-node=72 --gres=gpu:h100:1 --mem=200G --job-name=gpu-interactive --time 02:00:00 --pty bash -i
