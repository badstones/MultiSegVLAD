split=01
timesteps=10
num_centers=64
lr=0.02
dropout=0.8

first_step=80
second_step=150
total_epoch=210
two_steps=120
optim=SGD
prefix=ucf_slplit${split}
python ./main.py ucf101 RGB /root/data/ucf101_script/mo_train${split}.txt /root/data/ucf101_script/mo_test${split}.txt\
      --arch BNInception \
      --timesteps ${timesteps} --num_centers ${num_centers} --redu_dim 512 \
      --gd 20 --lr ${lr} --lr_steps ${first_step} ${second_step} --epochs ${total_epoch} \
      -b 32 -j 8 --dropout ${dropout} \
      --snapshot_pref ./models/rgb/${prefix} \
      --sources /root/data/UCF-101_rgb_flow\
      --two_steps ${two_steps} \
      --activation softmax \
      --optim ${optim}
