python export_model.py \
    --logtostderr \
    --checkpoint_path="/media/xzq/DA18EBFA09C1B27D/mobilenet/exp/train_on_train_set/train/model.ckpt-200000" \
    --export_path="/media/xzq/DA18EBFA09C1B27D/mobilenet/exp/train_on_train_set/export/frozen_inference_graph.pb" \
    --model_variant="mobilenet_v2" \
    --num_classes=19 \
    --crop_size=1024 \
    --crop_size=2048 \
    --output_stride=16 \
    --inference_scales=1.0