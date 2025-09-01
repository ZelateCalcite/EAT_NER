export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python train.py \
--epoch 50 \
--trd "train data path" \
--evd "dev data path" \
--eval True \
--cp "T5-base path" \
--on "output file .pt path"
