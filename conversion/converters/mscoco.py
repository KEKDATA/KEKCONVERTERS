import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict, DefaultDict, Union

from conversion.entities import KEKBox, KEKObject, KEKFormat


def construct_mscoco_dicts(coco_json_file_path: str) -> Tuple[Dict[str, Dict], DefaultDict[int, List], Dict[int, Dict]]:
    """
    Constructs necessary dicts from MS COCO JSON file for faster search.

    :param coco_json_file_path: Path to single MS COCO JSON file. Usually its a trainval2017.json.

    :return: Dictionaries with images, annotations and categories organized for faster search.
    """
    with open(coco_json_file_path, 'r') as jf:
        coco_json = json.load(jf)
    coco_images = {image['file_name']: image for image in coco_json['images']}
    coco_annotations = defaultdict(list)
    for annotation in coco_json['annotations']:
        coco_annotations[annotation['image_id']].append(annotation)
    coco_categories = {category['id']: category for category in coco_json['categories']}
    return coco_images, coco_annotations, coco_categories


def coco_annotations2kek_objects(annotations: List[Dict[str, Union[int, str, List[float]]]],
                                 coco_categories: Dict[int, Dict[str, Union[int, str]]]) -> List[KEKObject]:
    kek_objects = []
    for annotation in annotations:
        category_id = annotation['category_id']
        try:
            category = coco_categories[category_id]
        except KeyError:
            category = coco_categories[int(category_id)]
            category_id = int(category_id)
        ms_coco_metadata = {k: v for k, v in annotation.items() if k not in ('category_id', 'bbox')}
        kek_box = KEKBox.from_coco(annotation['bbox'])
        kek_objects.append(KEKObject(category_id - 1, category['name'], category.get('supercategory'), kek_box,
                                     ms_coco_metadata=ms_coco_metadata))
    return kek_objects


def mscoco2kek(image: os.DirEntry,
               base_annotation_path: str = None,
               hard: bool = True,
               coco_images: Dict[str, Dict[str, Union[int, str]]] = None,
               coco_annotations: Dict[int, List[Dict[str, Union[int, str, List[float]]]]] = None,
               coco_categories: Dict[int, Dict[str, Union[int, str]]] = None) -> KEKFormat:
    if hard:
        return mscoco_hard2kek(image, coco_images, coco_annotations, coco_categories)
    else:
        return mscoco_simple2kek(image, base_annotation_path, coco_categories)


def kek2mscoco(kek_format: KEKFormat,
               hard: bool = True) -> Union[Tuple[Dict[str, Union[int, str]],
                                                 List[Dict[str, Union[int, str, List[float]]]],
                                                 Dict[int, Dict[str, Union[int, str]]]],
                                           Dict[str, Union[List[Dict[str, Union[List[float], int]]], int]]]:
    if hard:
        return kek2mscoco_hard(kek_format)
    else:
        return kek2mscoco_simple(kek_format)


def kek2mscoco_simple(kek_format: KEKFormat) -> Dict[str, Union[List[Dict[str, Union[List[float], int]]], int]]:
    kek_objects, image_metadata = kek_format.kek_objects, kek_format.image_metadata
    json_file = dict.fromkeys(('annotation', 'image'))
    json_file['image'] = image_metadata
    annotations = []
    for kek_object in kek_objects:
        ms_coco_metadata = kek_object.mc_metadata
        ms_coco_metadata.update({'category_id': kek_object.class_id + 1,
                                 'bbox': kek_object.kek_box.to_coco_box()})
        annotations.append(ms_coco_metadata)
    json_file['annotation'] = annotations
    return json_file


def mscoco_simple2kek(image: os.DirEntry, base_annotation_path: str,
                      coco_categories: Dict[int, Dict[str, Union[int, str]]]) -> KEKFormat:
    image_name, image_ext = os.path.splitext(image.name)
    if not base_annotation_path:
        base_json_path = os.path.split(image.path)[0]
    else:
        base_json_path = base_annotation_path
    json_name = '.'.join([image_name, 'json'])
    json_path = os.path.join(base_json_path, json_name)
    with open(json_path, 'r') as jf:
        labels = json.load(jf)
    image_dict = labels['image']
    annotations = labels['annotation']
    kek_objects = coco_annotations2kek_objects(annotations, coco_categories)
    kek_format = KEKFormat(kek_objects, image_dict)
    return kek_format


def kek2mscoco_hard(kek_format: KEKFormat) -> Tuple[Dict[str, Union[int, str]],
                                                    List[Dict[str, Union[int, str, List[float]]]],
                                                    Dict[int, Dict[str, Union[int, str]]]]:
    kek_objects, image_metadata = kek_format.kek_objects, kek_format.image_metadata
    annotations = []
    categories = {}
    for kek_object in kek_objects:
        ms_coco_metadata = kek_object.mc_metadata
        ms_coco_metadata.update({'category_id': kek_object.class_id + 1,
                                 'bbox': kek_object.kek_box.to_coco_box()})
        annotations.append(ms_coco_metadata)
        category_id = kek_object.class_id + 1
        categories.update({category_id: {'supercategory': kek_object.superclass, 'id': category_id,
                                         'name': kek_object.class_name}})
    return kek_format.image_metadata, annotations, categories


def mscoco_hard2kek(image: os.DirEntry, coco_images: Dict[str, Dict[str, Union[int, str]]],
                    coco_annotations: Dict[int, List[Dict[str, Union[int, str, List[float]]]]],
                    coco_categories: Dict[int, Dict[str, Union[int, str]]]) -> KEKFormat:
    """Converts hard variant of MS COCO annotations. Hard MS COCO variant is variant where
    all information stores in one JSON file like source MS COCO trainval2017.json."""
    image_dict = coco_images[image.name]
    image_id = image_dict['id']
    annotations = coco_annotations[image_id]
    kek_objects = coco_annotations2kek_objects(annotations, coco_categories)
    kek_format = KEKFormat(kek_objects, image_dict)
    return kek_format
