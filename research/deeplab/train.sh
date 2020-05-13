python train.py \
    --logtostderr \
    --training_number_of_steps=1000 \
    --weight_decay=0.0001 \
    --multi_grid=1 \
    --multi_grid=2 \
    --multi_grid=4 \
    --train_split="train" \
    --model_variant="resnet_v1_101_beta" \
    --output_stride=16 \
    --train_crop_size[0]=769 \
    --train_crop_size[1]=769 \
    --train_batch_size=1 \
    --dataset="cityscapes" \
    --tf_initial_checkpoint='/home/xzq/models-master/research/deeplab/backbone/resnet_v1_101/model.ckpt' \
    --train_logdir='/media/xzq/DA18EBFA09C1B27D/resnet101/exp/train_on_train_set/train' \
    --dataset_dir='/home/xzq/data/cityscapes/tfrecord'

