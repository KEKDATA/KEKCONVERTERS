import os
import json
import random
import xml.etree.ElementTree as ET
from functools import partial
from collections import defaultdict

import pytest

import conversion.converters.mscoco as mc
import conversion.converters.darknet as dn
import conversion.converters.pascalvoc as pv
import conversion.converters.converters_utils as cu
from conversion.entities import KEKBox


def compare_pascal_voc_bounding_boxes(first_box: ET.Element,
                                      second_box: ET.Element,
                                      precision: int = 10) -> bool:
    try:
        numerical_first_box = [
            int(float(first_box.find('xmin').text)),
            int(float(first_box.find('xmax').text)),
            int(float(first_box.find('ymin').text)),
            int(float(first_box.find('ymax').text))
        ]
        numerical_second_box = [
            int(float(second_box.find('xmin').text)),
            int(float(second_box.find('xmax').text)),
            int(float(second_box.find('ymin').text)),
            int(float(second_box.find('ymax').text))
        ]
        first_and_second_boxes = zip(numerical_first_box, numerical_second_box)
        for first_box_coordinate, second_box_coordinate in \
                first_and_second_boxes:
            if abs(first_box_coordinate - second_box_coordinate) > precision:
                return False
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def compare_pascal_voc_objects(first_object: ET.Element,
                               second_object: ET.Element,
                               box_precision: int = 10) -> bool:
    try:
        first_object_name = first_object.find('name').text
        second_object_name = second_object.find('name').text
        if first_object_name != second_object_name:
            return False
        first_object_bndbox = first_object.find('bndbox')
        second_object_bndbox = second_object.find('bndbox')
        return compare_pascal_voc_bounding_boxes(
            first_object_bndbox,
            second_object_bndbox,
            box_precision
        )
    except (AttributeError, TypeError):
        return False


def compare_pascal_voc_annotations(first_annotation: ET.Element,
                                   second_annotation: ET.Element,
                                   box_precision: int = 10) -> bool:
    def object_sorter(object_: ET.Element) -> int:
        """IN GOD WE TRUST"""
        try:
            top_left_x = int(object_.find('bndbox').find('xmin').text)
        except ValueError:
            top_left_x = float(object_.find('bndbox').find('xmin').text)
        return top_left_x

    try:
        first_annotation_filename = first_annotation.find('filename').text
        second_annotation_filename = second_annotation.find('filename').text
        if first_annotation_filename != second_annotation_filename:
            return False
        first_annotation_objects = sorted(
            first_annotation.findall('object'),
            key=object_sorter
        )
        second_annotation_objects = sorted(
            second_annotation.findall('object'),
            key=object_sorter
        )
        objects_to_compare = zip(
            first_annotation_objects,
            second_annotation_objects
        )
        for first_object, second_object in objects_to_compare:
            return compare_pascal_voc_objects(
                first_object,
                second_object,
                box_precision
            )
    except (AttributeError, TypeError):
        return False


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
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'pascalvoc_and_darknet'
    )
    darknet_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'darknet',
        'source'
    )
    class_mapper_path = os.path.join(
        os.getcwd(),
        'test_data',
        'class_mappers',
        'darknet_mapper.json'
    )
    with open(class_mapper_path, 'r') as jf:
        class_mapper = json.load(jf)
    for image_id, image_name in enumerate(os.listdir(path_to_images)):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = dn.darknet2kek(
            image_path,
            image_id,
            class_mapper,
            darknet_annotation_path
        )
        converted_labels = dn.kek2darknet(kek_image)
        txt_path = cu.construct_annotation_file_path(
            image_path,
            'txt',
            darknet_annotation_path
        )
        with open(txt_path, 'r') as tf:
            true_labels = tf.readlines()
        for true_label, converted_label in zip(true_labels, converted_labels):
            assert compare_darknet_labels(true_label, converted_label)


def test_pascalvoc2pascalvoc():
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'pascalvoc_and_darknet'
    )
    pascalvoc_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'pascalvoc',
        'source'
    )
    class_mapper_path = os.path.join(
        os.getcwd(),
        'test_data',
        'class_mappers',
        'pascalvoc_mapper.json'
    )
    with open(class_mapper_path, 'r') as jf:
        class_mapper = json.load(jf)
    for image_id, image_name in enumerate(os.listdir(path_to_images)):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = pv.pascalvoc2kek(
            image_path,
            image_id,
            class_mapper,
            pascalvoc_annotation_path
        )
        converted_labels = pv.kek2pascalvoc(kek_image)
        xml_path = cu.construct_annotation_file_path(
            image_path,
            'xml',
            pascalvoc_annotation_path
        )
        with open(xml_path, 'r') as tf:
            true_labels_lines = set([line[:-1] for line in tf.readlines()])
        converted_labels_lines = set([line for line
                                      in converted_labels.split('\n') if line])
        for true_label_line in true_labels_lines:
            assert true_label_line in converted_labels_lines


def test_mscoco_simple2coco_simple():
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'coco',
        'coco_simple'
    )
    mscoco_annotation_path = os.path.join(
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
    for image_name in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = mc.mscoco2kek(
            image_path,
            mscoco_annotation_path,
            hard=False,
            coco_categories=categories
        )
        converted_annotation, converted_categories = mc.kek2mscoco(
            kek_image,
            hard=False
        )
        true_label_path = cu.construct_annotation_file_path(
            image_path,
            'json',
            mscoco_annotation_path
        )
        with open(true_label_path, 'r') as jf:
            true_annotation = json.load(jf)
        true_image_dict = true_annotation['image']
        converted_image_dict = converted_annotation['image']
        assert is_subdict(true_image_dict, converted_image_dict)
        true_annotations = {(anno['id'], anno['image_id']): anno for anno in
                            true_annotation['annotation']}
        converted_annotations = {(anno['id'], anno['image_id']): anno
                                 for anno in converted_annotation['annotation']}
        assert is_subdict(
            true_annotations,
            converted_annotations,
            comparator=partial(is_subdict, comparator=kek_comparator)
        )


def test_mscoco_hard2mscoco_hard():
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'coco',
        'coco_hard'
    )
    mscoco_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'single_file_mode',
        'single.json'
    )
    true_images, true_annotations, true_categories = mc.construct_mscoco_dicts(
        mscoco_annotation_path
    )
    converted_big_dict = {'images': [], 'annotations': [], 'categories': {}}
    for image_name in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = mc.mscoco2kek(
            image_path,
            None,
            True,
            true_images,
            true_annotations,
            true_categories
        )
        (converted_image_dict, converted_annotations,
         converted_categories) = mc.kek2mscoco(kek_image, hard=True)
        converted_big_dict['images'].append(converted_image_dict)
        converted_big_dict['annotations'].extend(converted_annotations)
        for category_id, category in converted_categories.items():
            converted_big_dict['categories'].update({category_id: category})
    converted_dict_of_images = {
        image_dict['file_name']: image_dict for image_dict in
        converted_big_dict['images']
    }
    assert is_subdict(
        true_images,
        converted_dict_of_images,
        comparator=partial(is_subdict, comparator=kek_comparator)
    )
    converted_dict_of_annotations = defaultdict(list)
    for annotation_dict in converted_big_dict['annotations']:
        converted_dict_of_annotations[annotation_dict['image_id']].append(
            annotation_dict
        )
    for image_id, true_annotations in true_annotations.items():
        converted_annotations = converted_dict_of_annotations[image_id]
        for true_annotation in true_annotations:
            converted_annotation = next(d for d in converted_annotations
                                        if d['id'] == true_annotation['id'])
            assert is_subdict(
                true_annotation,
                converted_annotation,
                kek_comparator
            )
    assert is_subdict(true_categories, converted_big_dict['categories'])


def test_pascalvoc2darknet():
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'pascalvoc_and_darknet'
    )
    pascalvoc_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'pascalvoc',
        'source'
    )

    # For comparison.
    darknet_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'darknet',
        'source'
    )
    pascalvoc_mapper_path = os.path.join(
        os.getcwd(),
        'test_data',
        'class_mappers',
        'pascalvoc_mapper.json'
    )
    with open(pascalvoc_mapper_path, 'r') as jf:
        class_mapper = json.load(jf)
    for image_id, image_name in enumerate(os.listdir(path_to_images)):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = pv.pascalvoc2kek(
            image_path,
            image_id,
            class_mapper,
            pascalvoc_annotation_path
        )
        converted_labels = dn.kek2darknet(kek_image)
        true_labels_path = cu.construct_annotation_file_path(
            image_path,
            'txt',
            darknet_annotation_path
        )
        with open(true_labels_path, 'r') as tf:
            true_labels = tf.readlines()
        for converted_label, true_label in zip(converted_labels, true_labels):
            assert compare_darknet_labels(converted_label, true_label)


def test_darknet2pascalvoc():
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'pascalvoc_and_darknet'
    )
    darknet_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'darknet',
        'source'
    )

    # For comparison.
    pascalvoc_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'pascalvoc',
        'source'
    )
    darknet_mapper_path = os.path.join(
        os.getcwd(),
        'test_data',
        'class_mappers',
        'darknet_mapper.json'
    )
    with open(darknet_mapper_path, 'r') as jf:
        class_mapper = json.load(jf)
    for image_id, image_name in enumerate(os.listdir(path_to_images)):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = dn.darknet2kek(
            image_path,
            image_id,
            class_mapper,
            darknet_annotation_path
        )
        converted_labels = ET.fromstring(pv.kek2pascalvoc(kek_image))
        true_labels_path = cu.construct_annotation_file_path(
            image_path,
            'xml',
            pascalvoc_annotation_path
        )
        true_labels = ET.parse(true_labels_path).getroot()
        assert compare_pascal_voc_annotations(true_labels, converted_labels)


def test_mscoco_simple2darknet():
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'coco',
        'coco_simple'
    )
    mscoco_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'multiple_files_mode'
    )
    mscoco_categories_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'multiple_files_mode',
        'catids.json'
    )

    # For comparison.
    darknet_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'darknet',
        'for_mscoco_simple_conversion_test'
    )
    with open(mscoco_categories_path, 'r') as jf:
        categories = {category['id']: category for category in json.load(jf)}
    for image_name in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = mc.mscoco2kek(
            image_path,
            mscoco_annotation_path,
            hard=False,
            coco_categories=categories
        )
        converted_labels = dn.kek2darknet(kek_image)
        true_labels_path = cu.construct_annotation_file_path(
            image_path,
            'txt',
            darknet_annotation_path
        )
        with open(true_labels_path, 'r') as tf:
            true_labels = tf.readlines()
        for true_label, converted_label in zip(true_labels, converted_labels):
            assert compare_darknet_labels(true_label, converted_label)


def test_mscoco_simple2pascalvoc():
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'coco',
        'coco_simple'
    )
    mscoco_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'multiple_files_mode'
    )
    mscoco_categories_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'multiple_files_mode',
        'catids.json'
    )

    # For comparison.
    pascalvoc_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'pascalvoc',
        'for_mscoco_simple_conversion_test'
    )
    with open(mscoco_categories_path, 'r') as jf:
        categories = {category['id']: category for category in json.load(jf)}
    for image_name in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = mc.mscoco2kek(
            image_path,
            mscoco_annotation_path,
            hard=False,
            coco_categories=categories
        )
        converted_labels = ET.fromstring(pv.kek2pascalvoc(kek_image))
        true_labels_path = cu.construct_annotation_file_path(
            image_path,
            'xml',
            pascalvoc_annotation_path
        )
        true_labels = ET.parse(true_labels_path).getroot()
        assert compare_pascal_voc_annotations(true_labels, converted_labels)


def test_mscoco_hard2pascalvoc():
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'coco',
        'coco_hard'
    )
    mscoco_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'single_file_mode',
        'single.json'
    )

    # For comparison.
    pascalvoc_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'pascalvoc',
        'for_mscoco_hard_conversion_test'
    )
    coco_images, coco_annotations, coco_categories = mc.construct_mscoco_dicts(
        mscoco_annotation_path
    )
    for image_name in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = mc.mscoco2kek(
            image_path,
            hard=True,
            coco_images=coco_images,
            coco_annotations=coco_annotations,
            coco_categories=coco_categories
        )
        converted_labels = ET.fromstring(pv.kek2pascalvoc(kek_image))
        true_labels_path = cu.construct_annotation_file_path(
            image_path,
            'xml',
            pascalvoc_annotation_path
        )
        true_labels = ET.parse(true_labels_path).getroot()
        assert compare_pascal_voc_annotations(true_labels, converted_labels)


def test_mscoco_hard2darknet():
    path_to_images = os.path.join(
        os.getcwd(),
        'test_data',
        'images',
        'coco',
        'coco_hard'
    )
    mscoco_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'mscoco',
        'single_file_mode',
        'single.json'
    )

    # For comparison.
    darknet_annotation_path = os.path.join(
        os.getcwd(),
        'test_data',
        'source_annotations',
        'darknet',
        'for_mscoco_hard_conversion_test'
    )
    coco_images, coco_annotations, coco_categories = mc.construct_mscoco_dicts(
        mscoco_annotation_path
    )
    for image_name in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_name)
        kek_image = mc.mscoco2kek(
            image_path,
            hard=True,
            coco_images=coco_images,
            coco_annotations=coco_annotations,
            coco_categories=coco_categories
        )
        converted_labels = dn.kek2darknet(kek_image)
        true_labels_path = cu.construct_annotation_file_path(
            image_path,
            'txt',
            darknet_annotation_path
        )
        with open(true_labels_path, 'r') as tf:
            true_labels = tf.readlines()
        for true_label, converted_label in zip(true_labels, converted_labels):
            assert compare_darknet_labels(true_label, converted_label)
