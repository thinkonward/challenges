from engine import Validator

if __name__ == "__main__":
    cfg_path = r"/root/autodl-tmp/Dark_side_of_the_volume/code/src/configs/mmseg_upernet_convnexts.py"
    root_dir = r"/root/autodl-tmp/Dark_side_of_the_volume/data/train_data_normed/"
    txt_file = r"/root/autodl-tmp/Dark_side_of_the_volume/data/train_txt/test_data.txt"
    model_path = r"/root/autodl-tmp/Dark_side_of_the_volume/exps/imagenet_norm_upernet_convnexts_bs16_hflip_amp_dice/best_model_score0.7788800992046876.pth"
    validator = Validator(cfg_path, model_path, root_dir, txt_file)
    score = validator.evaluate()
    print("score = ", score)