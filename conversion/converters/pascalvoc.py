import json
import os
import xml.etree.ElementTree as ET
from typing import Dict
from collections import defaultdict
from xml.dom.minidom import parseString

from conversion.utils import get_image_shape
from conversion.utils import warn_filename_not_found
from conversion.utils import construct_annotation_file_path
from conversion.utils import construct_additional_image_data
from conversion.utils import construct_additional_object_data
from conversion.entities import KEKBox, KEKObject, KEKImage


def _xml2dict(root):
    """

    :param root:
    :return:
    """
    TEXT_TOKEN = '#text'
    tree_dict = {root.tag: {} if root.attrib else None}
    children = list(root)
    if children:
        children_dict = defaultdict(list)
        for child_dict in map(_xml2dict, children):
            for key, value in child_dict.items():
                children_dict[key].append(value)
        tree_dict = {root.tag: {key: value[0] if len(value) == 1 else value
                                for key, value in children_dict.items()}}
    if root.text:
        text = root.text.strip()
        if children:
            if text:
                tree_dict[root.tag][TEXT_TOKEN] = text
        else:
            tree_dict[root.tag] = text
    return tree_dict


def _get_kek_box(object_element: ET.Element, xml_name: str) -> KEKBox:
    """
    Converts PASCAL VOC 'bndbox' tag content to KEKBox.

    :param object_element: XML element with PASCAL VOC 'object' tag;
    :param xml_name: Name of corresponding annotation .xml file.

    :raise ValueError if annotation .xml file has at least one object without
    bounding-box coordinate tag (<xmin>, <xmax>, <ymin>, <ymax>);
    :raise ValueError if annotation .xml file has at least one object with
    every necessary coordinate tags but some of these tags are empty.

    :return: KEKBox constructed from PASCAL VOC 'bndbox'.
    """
    bndbox = object_element.find('bndbox')
    if bndbox is None:
        raise ValueError('Annotation file {} has at least one object '
                         'without bounding-box.'.format(xml_name))
    xmin = bndbox.find('xmin')
    ymin = bndbox.find('ymin')
    xmax = bndbox.find('xmax')
    ymax = bndbox.find('ymax')
    if any((coordinate_tag is None
            for coordinate_tag in (xmin, ymin, xmax, ymax))):
        raise ValueError('Annotation file {} has at least one object '
                         'without coordinate tag.'.format(xml_name))
    top_left_x = xmin.text
    top_left_y = ymin.text
    bottom_right_x = xmax.text
    bottom_right_y = ymax.text
    if any((coordinate is None for coordinate in
            (top_left_x, top_left_y, bottom_right_x, bottom_right_y))):
        raise ValueError('Annotation file {} has at least one object '
                         'with empty coordinate tag.'.format(xml_name))
    return KEKBox.from_voc((int(top_left_x), int(top_left_y),
                            int(bottom_right_x), int(bottom_right_y)))


def _get_name(object_element: ET.Element, xml_name: str) -> str:
    """
    Converts PASCAL VOC 'bndbox''s 'name' tag content to string. Surprisingly.

    :param object_element: XML element with PASCAL VOC 'object' tag;
    :param xml_name: Name of corresponding annotation .xml file.

    :return: Class name string.
    """
    name = object_element.find('name')
    if name is None:
        raise ValueError('Annotation file {} has at least one object '
                         'without class name.'.format(xml_name))
    if not name.text:
        raise ValueError('Annotation file {} has at least one '
                         'object with empty <name></name> '
                         'tag.'.format(xml_name))
    return name.text


def pascalvoc2kek(image: os.DirEntry, image_id: int,
                  class_mapper: Dict[str, int],
                  base_annotation_path: str = None) -> KEKImage:
    """

    :param image:
    :param image_id:
    :param class_mapper:
    :param base_annotation_path:
    :return:
    """
    xml_path = construct_annotation_file_path(
        image,
        'xml',
        base_annotation_path)
    annotation = ET.parse(xml_path).getroot()

    # Necessary image data.
    filename = annotation.find('filename')
    if filename is None:
        warn_filename_not_found(os.path.split(xml_path)[-1])
        filename = image.name
    else:
        filename = filename.text
    size = annotation.find('size')
    try:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)
        image_shape = width, height, depth
    except (AttributeError, ValueError, TypeError):
        image_shape = get_image_shape(image)

    # These image tags should not be considered as additional data during
    # annotation tag parsing.
    main_image_tags = ('filename', 'size', 'object')

    # These object tags should not be considered as additional data during
    # object tag parsing.
    main_object_tags = ('name', 'bndbox')

    image_additional_data = construct_additional_image_data(image)
    kek_objects = []
    for element in annotation:
        # Additional image data.
        if element.tag not in main_image_tags:
            image_additional_data.update(_xml2dict(element))

        elif element.tag == 'object':
            # Necessary object data.
            class_name = _get_name(element, os.path.split(xml_path)[-1])
            class_id = class_mapper[class_name]
            kek_box = _get_kek_box(element, os.path.split(xml_path)[-1])

            # Additional object data.
            object_additional_data = construct_additional_object_data(image_id)
            for object_element in element:
                if object_element.tag not in main_object_tags:
                    object_additional_data.update(_xml2dict(object_element))

            kek_objects.append(KEKObject(class_id, class_name, kek_box,
                                         object_additional_data))

    return KEKImage(image_id, filename, image_shape, kek_objects,
                    image_additional_data)


def kek2pascalvoc(kek_image: KEKImage):
    """

    :param kek_image:
    :return:
    """
    def append_data_from_dict_to_xml(key, value, root):
        """

        :param key:
        :param value:
        :param root:
        :return:
        """
        TEXT_MARK = '#'
        if key.startswith(TEXT_MARK):
            root.text = value
        elif isinstance(value, str):
            sub = ET.SubElement(root, key)
            sub.text = str(value)
        elif isinstance(value, dict):
            sub_root = ET.SubElement(root, key)
            for sub_key, sub_value in value.items():
                append_data_from_dict_to_xml(sub_key, sub_value, sub_root)
        elif isinstance(value, list):
            for element in value:
                append_data_from_dict_to_xml(key, element, root)

    annotation = ET.Element('annotation')

    # Main image data.
    filename = ET.SubElement(annotation, 'filename')
    filename.text = kek_image.filename
    size = ET.SubElement(annotation, 'size')
    for dimension_name, dimension_value in zip(('width', 'height', 'depth'),
                                               kek_image.shape):
        dim = ET.SubElement(size, dimension_name)
        dim.text = str(dimension_value)

    # Additional image data.
    for k, v in kek_image.additional_data.items():
        append_data_from_dict_to_xml(k, v, annotation)

    for kek_object in kek_image.kek_objects:
        object_ = ET.SubElement(annotation, 'object')

        # Main object data.
        name = ET.SubElement(object_, 'name')
        name.text = kek_object.class_name
        bndbox = ET.SubElement(object_, 'bndbox')
        kek_box = kek_object.kek_box.to_voc_box()
        for coordinate_tag, coordinate in zip(('xmin', 'ymin', 'xmax', 'ymax'),
                                              kek_box):
            coordinate_element = ET.SubElement(bndbox, coordinate_tag)
            coordinate_element.text = str(coordinate)

        # Additional object data.
        for k, v in kek_object.additional_data.items():
            append_data_from_dict_to_xml(k, v, object_)

    # We don't need header with xml version.
    xml_string = '\n'.join(
        parseString(ET.tostring(annotation)).toprettyxml().split('\n')[1:])
    return xml_string


if __name__ == '__main__':
    path_to_pascalvoc_src = '/home/wammy/PycharmProjects/KEKCONVERTERS' \
                            '/test_data/conversion_src/pascalvoc'
    path_to_pascalvoc_dst = '/home/wammy/PycharmProjects/KEKCONVERTERS' \
                            '/test_data/conversion_results/pascalvoc'
    path_to_images = '/home/wammy/PycharmProjects/KEKCONVERTERS/test_data/images/pascalvoc_and_darknet'
    path_to_pascalvoc_mapper = \
        '/home/wammy/PycharmProjects/KEKCONVERTERS/test_data/class_mappers' \
        '/pascalvoc_mapper.json'

    with open(path_to_pascalvoc_mapper, 'r') as jf:
        class_mapper = json.load(jf)
    for id_, img in enumerate(os.scandir(path_to_images)):
        kekf = pascalvoc2kek(img, id_, class_mapper, path_to_pascalvoc_src)
        print(kek2pascalvoc(kekf))
        break
        # dl = kek2darknet(kekf)
        # txt_name = os.path.splitext(img.name)[0] + '.txt'
        # txt_path = os.path.join(path_to_pascalvoc_dst, txt_name)
        # with open(txt_path, 'w') as tf:
        #     tf.writelines(dl)