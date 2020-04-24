import os
import xml.etree.ElementTree as ET
from typing import Dict
from xml.dom.minidom import parseString

import conversion.converters.converters_utils as cu
from conversion.entities import KEKBox, KEKObject, KEKImage


def get_kek_box(object_element: ET.Element, xml_name: str) -> KEKBox:
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
        raise ValueError(
            'Annotation file {} has at least one object without '
            'bounding-box.'.format(xml_name)
        )
    xmin = bndbox.find('xmin')
    ymin = bndbox.find('ymin')
    xmax = bndbox.find('xmax')
    ymax = bndbox.find('ymax')
    if any((coordinate_tag is None
            for coordinate_tag in (xmin, ymin, xmax, ymax))):
        raise ValueError(
            'Annotation file {} has at least one object without coordinate '
            'tag.'.format(xml_name)
        )
    top_left_x = xmin.text
    top_left_y = ymin.text
    bottom_right_x = xmax.text
    bottom_right_y = ymax.text
    if any((coordinate is None for coordinate in
            (top_left_x, top_left_y, bottom_right_x, bottom_right_y))):
        raise ValueError(
            'Annotation file {} has at least one object with empty coordinate '
            'tag.'.format(xml_name)
        )
    return KEKBox.from_voc(
        (
            int(top_left_x),
            int(top_left_y),
            int(bottom_right_x),
            int(bottom_right_y)
        )
    )


def get_name(object_element: ET.Element, xml_name: str) -> str:
    """
    Converts PASCAL VOC 'bndbox''s 'name' tag content to string. Surprisingly.

    :param object_element: XML element with PASCAL VOC 'object' tag;
    :param xml_name: Name of corresponding annotation .xml file.

    :return: Class name string.
    """
    name = object_element.find('name')
    if name is None:
        raise ValueError(
            'Annotation file {} has at least one object without class '
            'name.'.format(xml_name)
        )
    if not name.text:
        raise ValueError(
            'Annotation file {} has at least one object with empty '
            '<name></name> tag.'.format(xml_name)
        )
    return name.text


def pascalvoc2kek(
        image_path: str,
        image_id: int,
        class_mapper: Dict[str, int],
        base_annotation_path: str = None
) -> KEKImage:
    xml_path = cu.construct_annotation_file_path(
        image_path,
        cu.get_target_annotation_file_extension('pascalvoc'),
        base_annotation_path
    )
    annotation = ET.parse(xml_path).getroot()

    # Necessary image data.
    filename = annotation.find('filename')
    if filename is None:
        cu.warn_filename_not_found(os.path.split(xml_path)[-1])
        filename = os.path.split(image_path)[-1]
    else:
        filename = filename.text
    size = annotation.find('size')
    try:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)
        image_shape = width, height, depth
    except (AttributeError, ValueError, TypeError):
        image_shape = cu.get_image_shape(image_path)

    # These image tags should not be considered as additional data during
    # annotation tag parsing.
    main_image_tags = ('filename', 'size', 'object')

    # These object tags should not be considered as additional data during
    # object tag parsing.
    main_object_tags = ('name', 'bndbox')
    image_additional_data = cu.construct_additional_image_data(image_path)
    kek_objects = []
    for element in annotation:

        # Additional image data.
        if element.tag not in main_image_tags:
            image_additional_data.update(cu.xml2dict(element))
        elif element.tag == 'object':

            # Necessary object data.
            class_name = get_name(element, os.path.split(xml_path)[-1])
            class_id = class_mapper[class_name]
            kek_box = get_kek_box(element, os.path.split(xml_path)[-1])

            # Additional object data.
            object_additional_data = cu.construct_additional_object_data(
                image_id
            )
            for object_element in element:
                if object_element.tag not in main_object_tags:
                    object_additional_data.update(cu.xml2dict(object_element))
            kek_objects.append(KEKObject(class_id, class_name, kek_box,
                                         object_additional_data))
    return KEKImage(
        image_id,
        filename,
        image_shape,
        kek_objects,
        image_additional_data
    )


def kek2pascalvoc(kek_image: KEKImage):
    annotation = ET.Element('annotation')

    # Main image data.
    cu.append_data_from_dict_to_xml('filename', kek_image.filename, annotation)
    cu.append_data_from_dict_to_xml(
        'size',
        dict(zip(('width', 'height', 'depth'), kek_image.shape)),
        annotation
    )

    # Additional image data.
    for k, v in kek_image.additional_data.items():
        cu.append_data_from_dict_to_xml(k, v, annotation)
    for kek_object in kek_image.kek_objects:
        object_ = ET.SubElement(annotation, 'object')

        # Main object data.
        cu.append_data_from_dict_to_xml('name', kek_object.class_name, object_)
        kek_box = kek_object.kek_box.to_voc_box()
        cu.append_data_from_dict_to_xml(
            'bndbox ',
            dict(zip(('xmin', 'ymin', 'xmax', 'ymax'), kek_box)),
            object_
        )

        # Additional object data.
        for k, v in kek_object.additional_data.items():
            cu.append_data_from_dict_to_xml(k, v, object_)

    # We don't need header with xml version.
    xml_string = '\n'.join(
        parseString(ET.tostring(annotation)).toprettyxml().split('\n')[1:]
    )
    return xml_string


def save_annotation(path: str, annotation: str) -> None:
    with open(path, 'w') as tf:
        tf.write(annotation)
