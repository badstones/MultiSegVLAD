split=01
timesteps=5
num_centers=64
lr=0.02
dropout=0.8

first_step=80
second_step=300
total_epoch=450
two_steps=180
optim=SGD

prefix=ucf101_flow_split${split}
python ./main.py ucf101 Flow /root/data/ucf101_script/mo_train${split}.txt /root/data/ucf101_script/mo_test${split}.txt\
      --arch BNInception \
      --timesteps ${timesteps} --num_centers ${num_centers} --redu_dim 512 \
      --gd 20 --lr ${lr} --lr_steps ${first_step} ${second_step} --epochs ${total_epoch} \
      -b 32 -j 8 --dropout ${dropout} \
      --snapshot_pref ./models/flow/${prefix} \
      --sources /root/data/UCF-101_rgb_flow\
      --two_steps ${two_steps} \
      --activation softmax \
      --optim ${optim}
