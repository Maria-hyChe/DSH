config = {
    "backbone": "mobilenet",  # 所使用的的主干网络：mobilenet / xception
    "num_classes": 4,  # 自己需要的分类个数
    "downsample_factor": 8,  # 下采样的倍数8、16 下采样的倍数较小、理论上效果更好，但也要求更大的显存
    "input_shape": [256, 256],  # 输入图片的大小
    "UnFreeze_Epoch": 123,  # 模型总共训练的epoch
    "Unfreeze_batch_size": 16,  # 模型在解冻后的batch_size
    "Freeze_Train": False,  # 是否进行冻结训练
    "Freeze_Epoch": 123,  # 模型冻结训练的Freeze_Epoch (当Freeze_Train=False时失效)
    "Freeze_batch_size": 64,  # 模型冻结训练的batch_size (当Freeze_Train=False时失效)
    "num_workers":0,
    "eval_flag": True,  # 是否在训练时进行评估，评估对象为验证集
    "eval_period": 5,  # 代表多少个epoch评估一次，不建议频繁的评估
    "save_dir": 'logs',  # 权值与日志文件保存的文件夹
    "save_period": 5,  # 多少个epoch保存一次权值
    "dataset_path": 'dataset',  # 数据集路径
    "source_image_path": "train_set/image_wdlc_july_10bands", # 影像文件路径
    "label_data_path": "train_set/label_wdlc_10bands",  # 标签文件路径
    "Image_txt_Sets": "train_set/ImageSets/Segmentation",  # 生成属于训练集、测试集的对象的文本文档
    "predicted_img_dir": "prediction_img/",  # 被预测的影像的文件夹
    "predicted_img_name": "s220220816_har_wdlc_35km.tif",  # 被预测的影像的名字
    "predicted_img_label_dir": "prediction_true_class/",  # 被预测的影像对应的label文件所在的文件夹
    "predicted_img_label_name": "c_label_2020.tif",  # 被预测的影像对应的label文件名
    "predicted_img_out_dir": "prediction_5-10/",  # 被预测的影像的预测结果文件夹
    "predicted_img_out_name": "1",  # 被预测的影像的预测结果的文件名
    "predicted_img_prb_out_name":"img_prb_2"
}

if __name__ == '__main__':
    print(config["backbone"])
