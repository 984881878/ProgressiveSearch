# ProgressiveSearch

the details of this project is explained in my paper 'Efficiently Layer-wise Progressive Neural Architecture Search'

requirement pytorch >= 1.0.


### search model architecture with the determined layer fixed(accuracy and loss weiht is respectively 0.8 0.2)

python cifar10_arch_search.py --share_classifier --train_batch_size 256 --test_batch_size 512 --pretrained_batch 256 --n_worker 16 --n_epochs 1 --pretrained_epochs 5 --fix_determined --determined_train --re_init "last_determined" --determined_train_epoch 5 --determined_train_batch 128 --depth 74 --alpha 48 --target_hardware "gpu1" --reg_loss_type "add" --reg_loss_acc 0.8 --reg_loss_latency 0.2


### search model architecture with the determined layer fixed(accuracy and loss weiht is respectively 0.9 0.1)

python cifar10_arch_search.py --share_classifier --train_batch_size 256 --test_batch_size 512 --pretrained_batch 256 --n_worker 16 --n_epochs 1 --pretrained_epochs 5 --fix_determined --determined_train --re_init "last_determined" --determined_train_epoch 5 --determined_train_batch 128 --depth 74 --alpha 48 --target_hardware "gpu1" --reg_loss_type "add" --reg_loss_acc 0.9 --reg_loss_latency 0.1

### search model architecture not fix determined layer(accuracy and loss weiht is respectively 0.8 0.2)

python cifar10_arch_search.py --share_classifier --train_batch_size 64 --test_batch_size 512 --pretrained_batch 256 --n_worker 16 --n_epochs 2 --pretrained_epochs 5 --determined_train --re_init "no_reinit" --determined_train_epoch 3 --determined_train_batch 128 --depth 56 --alpha 48 --target_hardware "gpu1" --reg_loss_type "add" --reg_loss_acc 0.8 --reg_loss_latency 0.2

### search model architecture not fix the determined layer(accuracy and loss weiht is respectively 0.9 0.1)

python cifar10_arch_search.py --share_classifier --train_batch_size 64 --test_batch_size 512 --pretrained_batch 256 --n_worker 16 --n_epochs 2 --pretrained_epochs 5 --determined_train --re_init "no_reinit" --determined_train_epoch 3 --determined_train_batch 128 --depth 56 --alpha 48 --target_hardware "gpu1" --reg_loss_type "add" --reg_loss_acc 0.9 --reg_loss_latency 0.1
