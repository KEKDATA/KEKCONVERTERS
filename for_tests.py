from conversion.kek_format import KEKBox

IMAGE_WIDTH = 500
IMAGE_HEIGHT = 500
IMAGE_DEPTH = 3

PASCAL_VOC_BOX = [50, 50, 300, 300]
MS_COCO_BOX = [50, 50, 250, 250]
DARKNET_BOX = [0, 7. / 20, 7. / 20, 3. / 5, 3. / 5]
DARKNET_BOX_STR = ' '.join(map(str, DARKNET_BOX))


if __name__ == '__main__':
    kek_box = KEKBox([192, 168, 1, 167])
    print(kek_box)

    from_darknet = KEKBox.from_darknet(DARKNET_BOX, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    to_darknet = from_darknet.to_darknet_box((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    print(DARKNET_BOX)
    print(to_darknet)

    from_darknet_string = KEKBox.from_darknet(DARKNET_BOX_STR, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    to_darknet = from_darknet.to_darknet_box((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    print(DARKNET_BOX_STR)
    print(to_darknet)

    from_voc = KEKBox.from_voc(PASCAL_VOC_BOX)
    to_voc = from_voc.to_voc_box()
    print(PASCAL_VOC_BOX)
    print(to_voc)

    from_coco = KEKBox.from_coco(MS_COCO_BOX)
    to_coco = from_coco.to_coco_box()
    print(MS_COCO_BOX)
    print(to_coco)
