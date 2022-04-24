# Video-Person-Search
基于视频的行人检索


## 检测视频中的行人并每隔100帧保存图像到bbox_datasets文件中
```

python yolo_detection/object_detection_yolo.py

```



## 训练重识别模型

```

python feature_extract/train.py

```


## 提取查询图像和视频中检测出的行人图像特征保存为npy文件

```

python feature_extract/npy_extract.py

```

## 查询排序并获取Rank-10的结果，存为图像

```

/ssd/wwz/cv/bishe/lyl_Person_Reid/reid_query.py

```

## 视频检索行人，并保存结果

```

python3 object_detection_yolo.py --video=run.mp4 --device 'gpu'

```


![排序结果](https://github.com/amazingcodeLYL/Video-Person-Search/blob/master/4_ft10.png)

## 训练权重：
链接：https://pan.baidu.com/s/1W7T-0zkVsa5B-TRblHhuIA 
提取码：eslk




