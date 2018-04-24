server=$SIXNIGHT
dataset_id=4
dataset_part=cloth${dataset_id}
echo $dataset_id
echo $dataset_part
#hist,checkpoint,database,ssd1,ssd for color,lst file,
#checkpoint 除了上装都在68上
scp -r lhw@\[$SIXNIGHT\]:~/cloth/mxnet-classify/retrieval/clothall/$dataset_part/densenet201/densenet201_1920/cropus.npy ./retrieval/cropus/database/$dataset_id/cropus1920.npy
scp -r lhw@\[$SIXNIGHT\]:/home/lhw/cloth/mxnet-classify/data/hist/clothall/cloth4/cropus_hist.npy ./retrieval/cropus/hist/$dataset_id
#scp -r lhw@\[$SIXNIGHT\]:/home/lhw/cloth/mxnet-classify/checkpoint/clothall/cloth4/densenet201/net_best.params ./retrieval/checkpoint/$dataset_id
