from sahi.utils.coco import Coco
coco = Coco.from_coco_dict_or_path("/data/hyt/mmdetection/QZ/annotations/instances_train.json")
l = len(coco.images)
for i in range(l):
    coco_image = coco.images[i]
    for annotation in coco_image.annotations:
        if len(annotation.bbox) != 4:
            print(coco_image.annotations)
            print(annotation)
            print(coco_image)
        x, y, w, h = annotation.bbox
    
