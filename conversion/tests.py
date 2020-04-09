import os
import json
import random
from functools import partial
from collections import defaultdict

import pytest

import conversion.converters.mscoco as mc
import conversion.converters.darknet as dn
import conversion.converters.pascalvoc as pv
from conversion.entities import KEKBox
from conversion.utils import construct_annotation_file_path


def compare_darknet_labels(first_label: str, second_label: str,
                           precision: float = 10e-2) -> bool:
    numerical_first_label = map(float, first_label.split(' '))
    numerical_second_label = map(float, second_label.split(' '))
    first_and_second_labels = zip(numerical_first_label, numerical_second_label)
    for first_label_element, second_label_element in first_and_second_labels:
        if abs(first_label_element - second_label_element) > precision:
            return False
    return True


def is_subdict(subdict: dict, dict_: dict,
               comparator=lambda left, right: left == right) -> bool:
    try:
        for key in subdict.keys():
            if not comparator(subdict[key], dict_[key]):
                print(key, subdict[key])
                print(dict_[key])
                return False
        return True
    except KeyError:
        return False


def kek_comparator(src_value, dst_value) -> bool:
    if type(dst_value) is tuple and type(src_value) is list:
        if isinstance(dst_value[0], float):
            return all(abs(dv - sv) <= 0.01 for dv, sv in
                       zip(dst_value, src_value))
        return list(dst_value) == src_value
    if type(dst_value) is int and type(src_value) is str:
        return str(dst_value) == src_value
    return dst_value == src_value


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
            assert compare_darknet_labels(src_line, dst_line)


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
        for src_xml_line in src_xml_lines:
            assert src_xml_line in dst_xml_lines


def test_mscoco_simple2coco_simple():
    image_path = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'coco',
        'coco_simple'
    )
    source_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'multiple_files_mode'
    )
    category_ids_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'multiple_files_mode',
        'catids.json'
    )

    with open(category_ids_path, 'r') as jf:
        categories = json.load(jf)
        categories = {category['id']: category for category in categories}
    for img in os.scandir(image_path):
        kek_image = mc.mscoco2kek(
            img,
            source_annotation_path,
            hard=False,
            coco_categories=categories
        )
        dst_annotation, cats = mc.kek2mscoco(kek_image, hard=False)
        src_annotation_path = construct_annotation_file_path(
            img,
            'json',
            source_annotation_path
        )
        with open(src_annotation_path, 'r') as jf:
            src_annotation = json.load(jf)
        src_image_dict = src_annotation['image']
        dst_image_dict = dst_annotation['image']
        assert is_subdict(src_image_dict, dst_image_dict)
        src_annotations = {(anno['id'], anno['image_id']): anno for anno in
                           src_annotation['annotation']}
        dst_annotations = {(anno['id'], anno['image_id']): anno for anno in
                           dst_annotation['annotation']}
        assert is_subdict(
            src_annotations,
            dst_annotations,
            comparator=partial(is_subdict, comparator=kek_comparator)
        )


def test_mscoco_hard2mscoco_hard():
    image_path = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'coco',
        'coco_hard'
    )
    source_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'single_file_mode',
        'single.json'
    )
    coco_images, coco_annotations, coco_categories = mc.construct_mscoco_dicts(
        source_annotation_path
    )
    mscoco_big_dict = {'images': [], 'annotations': [], 'categories': {}}
    for img in os.scandir(image_path):
        kek_image = mc.mscoco2kek(
            img,
            None,
            True,
            coco_images,
            coco_annotations,
            coco_categories
        )
        image_dict, image_annotations, image_categories = mc.kek2mscoco(
            kek_image,
            hard=True
        )
        mscoco_big_dict['images'].append(image_dict)
        mscoco_big_dict['annotations'].extend(image_annotations)
        for category_id, category in image_categories.items():
            mscoco_big_dict['categories'].update({category_id: category})
    dst_dict_of_images = {
        image_dict['file_name']: image_dict for image_dict in
        mscoco_big_dict['images']
    }
    assert is_subdict(
        coco_images,
        dst_dict_of_images,
        comparator=partial(is_subdict, comparator=kek_comparator)
    )
    dst_dict_of_annotations = defaultdict(list)
    for annotation_dict in mscoco_big_dict['annotations']:
        dst_dict_of_annotations[annotation_dict['image_id']].append(
            annotation_dict
        )
    for image_id, annotations in coco_annotations.items():
        dst_annotations = dst_dict_of_annotations[image_id]
        for src_annotation in annotations:
            t = next(d for d in dst_annotations if d['id'] == src_annotation[
                'id'])
            assert is_subdict(src_annotation, t, kek_comparator)
    assert is_subdict(coco_categories, mscoco_big_dict['categories'])


def test_pascalvoc2darknet():
    image_path = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'pascalvoc_and_darknet'
    )
    pascalvoc_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'pascalvoc'
    )

    # For comparison.
    true_darknet_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'darknet'
    )
    pascalvoc_mapper_path = os.path.join(
        os.getcwd(),
        'test_data',
        'class_mappers',
        'pascalvoc_mapper.json'
    )
    with open(pascalvoc_mapper_path, 'r') as jf:
        class_mapper = json.load(jf)
    for image_id, img in enumerate(os.scandir(image_path)):
        kek_image = pv.pascalvoc2kek(
            img,
            image_id,
            class_mapper,
            pascalvoc_annotation_path
        )
        converted_darknet_labels = dn.kek2darknet(kek_image)
        true_darknet_txt_path = construct_annotation_file_path(
            img,
            'txt',
            true_darknet_annotation_path
        )
        with open(true_darknet_txt_path, 'r') as tf:
            true_darknet_labels = tf.readlines()
        for converted_label, true_label in zip(converted_darknet_labels,
                                               true_darknet_labels):
            assert compare_darknet_labels(converted_label, true_label)
