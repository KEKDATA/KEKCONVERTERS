import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict, DefaultDict, Union

from conversion.utils import get_image_shape
from conversion.utils import warn_filename_not_found
from conversion.utils import construct_annotation_file_path
from conversion.utils import construct_additional_image_data
from conversion.utils import construct_additional_object_data
from conversion.entities import KEKBox, KEKObject, KEKImage


def construct_mscoco_dicts(
        coco_json_file_path: str) -> Tuple[Dict[str, Dict], 
                                           DefaultDict[int, List],
                                           Dict[int, Dict]]:
    """
    Constructs necessary dicts from MS COCO JSON file for faster search.

    :param coco_json_file_path: Path to single MS COCO JSON file. Usually its a 
                                trainval2017.json.

    :return: Dictionaries with images, annotations and categories organized for 
             faster search.
    """
    with open(coco_json_file_path, 'r') as jf:
        coco_json = json.load(jf)
    coco_images = {image['file_name']: image for image in coco_json['images']}
    coco_annotations = defaultdict(list)
    for annotation in coco_json['annotations']:
        coco_annotations[annotation['image_id']].append(annotation)
    coco_categories = {category['id']: category for category in coco_json['categories']}
    return coco_images, coco_annotations, coco_categories


def coco_annotations2kek_objects(
        image_id: int,
        annotations: List[Dict[str, Union[int, str, List[float]]]],
        coco_categories: Dict[int,
                              Dict[str, Union[int, str]]]) -> List[KEKObject]:
    kek_objects = []
    for annotation in annotations:
        # Main object data.
        class_id = annotation['category_id']
        try:
            category = coco_categories[class_id]
        except KeyError:
            # Some smart guys used string category_id in
            # source annotation file.
            category = coco_categories[int(class_id)]
            class_id = int(class_id) - 1
        class_name = category['name']
        kek_box = KEKBox.from_coco(annotation['bbox'])

        # Additional object data.
        # These keys should not be processed during object data dictionary
        # parsing.
        main_object_data_keys = ('category_id', 'bbox',)
        object_additional_data = construct_additional_object_data(image_id)
        for key, value in annotation.items():
            if key not in main_object_data_keys:
                object_additional_data.update({key: value})
        kek_objects.append(KEKObject(class_id, class_name, kek_box,
                                     object_additional_data))

    return kek_objects


def mscoco2kek(image: os.DirEntry,
               base_annotation_path: str = None,
               hard: bool = True,
               coco_images: Dict[str, Dict[str, Union[int, str]]] = None,
               coco_annotations: Dict[int, List[Dict[str, Union[int, str, List[float]]]]] = None,
               coco_categories: Dict[int, Dict[str, Union[int, str]]] = None) -> KEKImage:
    if hard:
        return mscoco_hard2kek(image, coco_images, coco_annotations, coco_categories)
    else:
        return mscoco_simple2kek(image, base_annotation_path, coco_categories)


def kek2mscoco(kek_format: KEKImage,
               hard: bool = False) -> Union[Tuple[Dict[str, Union[int, str]],
                                                  List[Dict[str, Union[int, str, List[float]]]],
                                                  Dict[int, Dict[str, Union[int, str]]]],
                                            Dict[str, Union[List[Dict[str, Union[List[float], int]]], int]]]:
    if hard:
        return kek2mscoco_hard(kek_format)
    else:
        return kek2mscoco_simple(kek_format)


def kek2mscoco_simple(kek_image: KEKImage) -> Dict[str, Union[List[Dict[str,
                                                                    Union[List[float], int]]], int]]:
    json_file = dict.fromkeys(('annotation', 'image'))
    categories = {}

    # Image data.
    image_dict = kek_image.additional_data
    image_dict.update({'file_name': kek_image.filename})
    width, height, _ = kek_image.shape
    image_dict.update({'width': kek_image.filename, 'height': height})
    image_dict.update({'id': kek_image.id_})
    json_file['image'] = image_dict

    # Objects data.
    annotations = []
    for kek_object in kek_image.kek_objects:
        ms_coco_metadata = kek_object.additional_data
        ms_coco_metadata.update({'category_id': kek_object.class_id + 1,
                                 'bbox': kek_object.kek_box.to_coco_box()})
        annotations.append(ms_coco_metadata)
        categories.update(
            {kek_object.class_id + 1: {
                'category_id': kek_object.class_id + 1,
                'name': kek_object.class_name
            }})
    json_file['annotation'] = annotations
    return json_file, categories


def mscoco_simple2kek(image: os.DirEntry, base_annotation_path: str,
                      coco_categories: Dict[int, Dict[str, Union[int, str]]]) -> KEKImage:
    json_path = construct_annotation_file_path(image, 'json',
                                               base_annotation_path)
    with open(json_path, 'r') as jf:
        labels = json.load(jf)

    # Main image data.
    image_dict = labels['image']
    image_id = image_dict['id']
    try:
        filename = image_dict['file_name']
    except KeyError:
        # FIXME:
        warn_filename_not_found('HZ')
        filename = image.name
    try:
        width = image_dict['width']
        height = image_dict['height']
        depth = get_image_shape(image)[-1]
        image_shape = (width, height, depth)
    except KeyError:
        image_shape = get_image_shape(image)

    # Additional image data.
    # This keys should not be processed for additional image data.
    main_image_data_keys = ('id', 'file_name', 'width', 'height')
    image_additional_data = construct_additional_image_data(image)
    for key, value in image_dict:
        if key not in main_image_data_keys:
            image_additional_data.update({key: value})

    annotations = labels['annotation']
    kek_objects = coco_annotations2kek_objects(image_id, annotations,
                                               coco_categories)

    return KEKImage(image_id, filename, image_shape, kek_objects,
                    image_additional_data)


def kek2mscoco_hard(kek_format: KEKImage) -> Tuple[Dict[str, Union[int, str]],
                                                    List[Dict[str, Union[int, str, List[float]]]],
                                                    Dict[int, Dict[str, Union[int, str]]]]:
    # Image data.
    image_dict = kek_format.additional_data
    image_dict.update({'id': kek_format.id_})
    image_dict.update({'file_name': kek_format.filename})
    width, height, _ = kek_format.shape
    image_dict.update({'width': width, 'height': height})

    annotations = []
    categories = {}
    for kek_object in kek_format.kek_objects:
        # Object data.
        ms_coco_metadata = kek_object.additional_data
        ms_coco_metadata.update({'category_id': kek_object.class_id + 1,
                                 'bbox': kek_object.kek_box.to_coco_box()})
        annotations.append(ms_coco_metadata)

        # Categories.
        # Can't get supercategory information from other annotation formats.
        # Solve it?
        category_id = kek_object.class_id + 1
        categories.update({category_id: {'id': category_id,
                                         'name':kek_object.class_name}})

    return image_dict, annotations, categories


def mscoco_hard2kek(image: os.DirEntry, 
                    coco_images: Dict[str, Dict[str, Union[int, str]]],
                    coco_annotations: Dict[int, List[Dict[str, Union[int, str, List[float]]]]],
                    coco_categories: Dict[int, Dict[str, Union[int, str]]]) -> KEKImage:
    """Converts hard variant of MS COCO annotations. Hard MS COCO variant is variant where
    all information stores in one JSON file like source MS COCO trainval2017.json."""
    image_dict = coco_images[image.name]

    # Main image data.
    image_id = image_dict['id']
    try:
        filename = image_dict['file_name']
    except KeyError:
        # FIXME:
        warn_filename_not_found('HZ')
        filename = image.name
    try:
        width = image_dict['width']
        height = image_dict['height']
        depth = get_image_shape(image)[-1]
        image_shape = (width, height, depth)
    except KeyError:
        image_shape = get_image_shape(image)

    # Additional image data.
    # This keys should not be processed for additional image data.
    main_image_data_keys = ('id', 'file_name', 'width', 'height')
    image_additional_data = construct_additional_image_data(image)
    for key, value in image_dict.items():
        if key not in main_image_data_keys:
            image_additional_data.update({key: value})

    annotations = coco_annotations[image_id]
    kek_objects = coco_annotations2kek_objects(image_id, annotations,
                                               coco_categories)

    return KEKImage(image_id, filename, image_shape, kek_objects,
                    image_additional_data)
