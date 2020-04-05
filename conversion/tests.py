import os
import json
import random

import pytest

import conversion.converters.darknet as dn
import conversion.converters.pascalvoc as pv
from conversion.entities import KEKBox
from conversion.utils import construct_annotation_file_path


def test_kek_box():
    boxes = [
        (random.randint(0, 608), random.randint(0, 608), random.randint(0, 608),
         random.randint(0, 608)) for _ in range(10)]
    kek_boxes = [KEKBox(*box) for box in boxes]
    for box, kek_box in zip(boxes, kek_boxes):
        assert box == (kek_box.top_left_x, kek_box.top_left_y,
                       kek_box.bottom_right_x, kek_box.bottom_right_y)


def test_darknet2darknet():
    image_path = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'pascalvoc_and_darknet'
    )
    source_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'darknet'
    )
    class_mapper_path = os.path.join(
        os.getcwd(),
        'test_data',
        'class_mappers',
        'darknet_mapper.json'
    )
    with open(class_mapper_path, 'r') as jf:
        class_mapper = json.load(jf)
        class_mapper = {int(k): v for k, v in class_mapper.items()}
    for img_id, img in enumerate(os.scandir(image_path)):
        kek_image = dn.darknet2kek(img, img_id, class_mapper,
                                   source_annotation_path)
        dst_darknet_lines = dn.kek2darknet(kek_image)
        txt_path = construct_annotation_file_path(
            img, 'txt', source_annotation_path)
        with open(txt_path, 'r') as tf:
            src_darknet_lines = tf.readlines()
        for src_line, dst_line in zip(src_darknet_lines, dst_darknet_lines):
            (src_class_id, src_center_x, src_center_y, src_box_width,
             src_box_height) = map(float, src_line.split(' '))
            (dst_class_id, dst_center_x, dst_center_y, dst_box_width,
             dst_box_height) = map(float, dst_line.split(' '))
            assert src_class_id == dst_class_id
            assert abs(src_center_x - dst_center_x) < 10e-2
            assert abs(src_center_y - dst_center_y) < 10e-2
            assert abs(src_box_width - dst_box_width) < 10e-2
            assert abs(src_box_height - dst_box_height) < 10e-2


def test_pascalvoc2pascalvoc():
    image_path = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'pascalvoc_and_darknet'
    )
    source_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'pascalvoc'
    )
    class_mapper_path = os.path.join(
        os.getcwd(),
        'test_data',
        'class_mappers',
        'pascalvoc_mapper.json'
    )
    with open(class_mapper_path, 'r') as jf:
        class_mapper = json.load(jf)
    for img_id, img in enumerate(os.scandir(image_path)):
        kek_image = pv.pascalvoc2kek(
            img, img_id, class_mapper, source_annotation_path)
        dst_xml_string = pv.kek2pascalvoc(kek_image)
        xml_path = construct_annotation_file_path(
            img, 'xml', source_annotation_path)
        with open(xml_path, 'r') as tf:
            src_xml_lines = set([line[:-1] for line in tf.readlines()])
        dst_xml_lines = set([line for line in dst_xml_string.split('\n')
                             if line])
        assert dst_xml_lines == src_xml_lines
