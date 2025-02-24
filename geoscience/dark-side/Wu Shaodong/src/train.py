from engine import Trainer

if __name__ == "__main__":
    cfg_path = r"./src/configs/train_config_f0.py"
    trainer = Trainer(cfg_path)
    trainer.train()

    cfg_path = r"./src/configs/train_config_f1.py"
    trainer = Trainer(cfg_path)
    trainer.train()