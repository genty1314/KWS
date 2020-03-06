# coding=utf-8


def default_config():
    config = {}
    # model set
    config["group_speakers_by_id"] = True
    config["input_file"] = None   # model\EdgeCRNN-2x.pt
    config["n_labels"] = 12
    config["no_cuda"] = True

    # input shape
    config["silence_prob"] = 0.1
    config["noise_prob"] = 0.8
    config["n_dct_filters"] = 40
    config["input_length"] = 16000
    config["n_mels"] = 13  # MFCC-》39， log_mel->13 PCEN
    config["timeshift_ms"] = 100
    config["unknown_prob"] = 0.1
    config["train_pct"] = 80
    config["dev_pct"] = 10
    config["test_pct"] = 10
    config["wanted_words"] = ["zero", "one ", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    config["data_folder"] = "E:\Project-Data\honk\speech_commands_v0.02"
    config["audio_preprocess_type"] = "MFCCs"
    config["feature_type"] = "log_mel"  # ["MFCC", "log_mel", PCEN]

    # train parameter
    config["n_epochs"] = 500
    config["add_noise"] = True
    config["type"] = "train"  # [train, eval]
    config["loss"] = "focal"  # ["CE", "focal"]
    config["optimizer"] = "adam"  # ["adam", "sgd"]
    config["model_type"] = "EdgeCRNN"  # ["EdgeCRNN","shuffleNet", "Tpool2",
    #  "rnn", "mobileNet", "mobileNetV3-Small", "mobileNetV3-Large"]
    config["preprocess_data"] = 2  # [1, 2]  2 online, 1 offline
    config["width_mult"] = 1
    config["output_file"] = "model/EdgeCRNN"  #

    return config
