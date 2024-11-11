CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=0 python main.py \
--epochs 30 \
--batch_size 8 \
--target 'human_text_jamo' \
--asr_mode 'human' \
--train_filename 'r08.1_train.csv' \
--test_filename 'r08.1_test copy.csv' \
# --loss_feature 'target_text_id' \
# --dropout '0.2:6 7 8 9 10 11'