import os
import warnings
from typing import Tuple, Dict, Any
from collections import defaultdict
from xml.etree import ElementTree as ET

from PIL import Image


def warn_filename_not_found(annotation_file_name: str) -> None:
    """We should warn user if annotations doesn't contain filename in annotation
    file. We can get this filename from corresponding annotation file. And if
    this we should warn users about it."""
    warnings.warn(
        'Annotation file {} has no filename tag.'
        '\nGetting filename from file...'.format(annotation_file_name)
    )


def construct_annotation_file_path(
        image_path: str,
        annotation_file_extension: str,
        base_annotation_path: str = None
) -> str:
    image_name, image_ext = os.path.splitext(os.path.split(image_path)[-1])
    if not base_annotation_path:
        base_file_path = os.path.split(image_path)[0]
    else:
        base_file_path = base_annotation_path
    annotation_filename = ''.join([image_name, annotation_file_extension])
    return os.path.join(base_file_path, annotation_filename)


def get_image_shape(image_path: str) -> Tuple[int, int, int]:
    pil_image = Image.open(image_path)
    width, height = pil_image.size
    depth = len(pil_image.getbands())
    return width, height, depth


def construct_additional_image_data(image_path: str) -> Dict[str, str]:
    folder = os.path.split(os.path.split(image_path)[0])[-1]
    image_additional_data = {'path': image_path, 'folder': folder}
    return image_additional_data


def construct_additional_object_data(image_id: int) -> Dict[str, int]:
    return {'image_id': image_id}


def get_target_annotation_file_extension(target_annotation_name: str) -> str:
    return {
        'darknet': '.txt',
        'pascalvoc': '.xml',
        'mscoco': '.json'
    }.get(target_annotation_name)


def xml2dict(root: ET.Element) -> Dict[str, Any]:
    """Converts ElementTree.Element to dictionary in a recursive way."""
    TEXT_TOKEN = '#text'
    tree_dict = {root.tag: {} if root.attrib else None}
    children = list(root)
    if children:
        children_dict = defaultdict(list)
        for child_dict in map(xml2dict, children):
            for key, value in child_dict.items():
                children_dict[key].append(value)
        tree_dict = {
            root.tag: {
                key: value[0] if len(value) == 1 else value
                for key, value in children_dict.items()
            }
        }
    if root.text:
        text = root.text.strip()
        if not children:
            tree_dict[root.tag] = text
        elif text:
            tree_dict[root.tag][TEXT_TOKEN] = text
    return tree_dict


def append_data_from_dict_to_xml(
        key: str,
        value: Any,
        root: ET.Element
) -> None:
    """Appends data from (key, valye) pair to ElementTree element in a
    recursive way."""
    TEXT_MARK = '#'
    if key.startswith(TEXT_MARK):
        root.text = value
    elif (isinstance(value, str) or isinstance(value, float) or
          isinstance(value, int)):
        sub = ET.SubElement(root, key)
        sub.text = str(value)
    elif isinstance(value, dict):
        sub_root = ET.SubElement(root, key)
        for sub_key, sub_value in value.items():
            append_data_from_dict_to_xml(sub_key, sub_value, sub_root)
    elif isinstance(value, list):
        for element in value:
            append_data_from_dict_to_xml(key, element, root)
