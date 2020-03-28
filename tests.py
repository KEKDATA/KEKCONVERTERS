import pytest

from conversion.kek_format import KEKBox

IMAGE_WIDTH = 500
IMAGE_HEIGHT = 500
IMAGE_DEPTH = 3

PASCAL_VOC_BOX = (50, 50, 300, 300)
MS_COCO_BOX = (50, 50, 250, 250)
DARKNET_BOX = (7. / 20, 7. / 20, 3. / 5, 3. / 5)
DARKNET_BOX_STR = ' '.join(map(str, DARKNET_BOX))


def test_kek_box():
    kek_box = KEKBox([192, 168, 1, 167])
    assert str(kek_box) == '(192, 168, 1, 167)'


def test_darknet_box():
    from_darknet = KEKBox.from_darknet(DARKNET_BOX, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    to_darknet = from_darknet.to_darknet_box((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    assert to_darknet == DARKNET_BOX


def test_darknet_str_box():
    from_darknet = KEKBox.from_darknet(DARKNET_BOX_STR, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    str_result = ' '.join(map(str, from_darknet.to_darknet_box((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))))
    assert str_result == DARKNET_BOX_STR


def test_voc_box():
    from_voc = KEKBox.from_voc(PASCAL_VOC_BOX)
    to_voc = from_voc.to_voc_box()
    assert to_voc == PASCAL_VOC_BOX


def test_coco_box():
    from_coco = KEKBox.from_coco(MS_COCO_BOX)
    to_coco = from_coco.to_coco_box()
    assert to_coco == MS_COCO_BOX