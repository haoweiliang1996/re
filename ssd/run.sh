#python3 demo.py --gpu 0 --network resnet50 --epoch 175  --output './data/demo_out'   --dir '/home/lhw/cloth/triplet/type1/6/60' --image '4373.jpg,8266.jpg' --thresh 0.7 --prefix model/ssd --class-name 'skirt' --nms 0.5
python3 demo.py --gpu 0 --network resnet50 --epoch 175  --output './data/demo_out/demoout'   --dir '/home/lhw/cloth/triplet/type3/6/60' --image '4373.jpg,8266.jpg' --thresh 0.7 --prefix model/ssd --class-name 'skirt' --nms 0.5

