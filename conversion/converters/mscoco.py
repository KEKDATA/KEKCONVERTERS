import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict, DefaultDict, Union, Any

import conversion.converters.converters_utils as cu
from conversion.entities import KEKBox, KEKObject, KEKImage


def construct_mscoco_dicts(coco_json_file_path: str) -> \
        Tuple[
            Dict[str, Dict[str, Any]],
            DefaultDict[int, List[Dict[str, Any]]],
            Dict[int, Dict[str, Any]]
        ]:
    """
    Constructs necessary dicts from MS COCO JSON file for faster search.

    :param coco_json_file_path: Path to single MS COCO JSON file.

    :return: Dictionaries with images, annotations and categories organized for 
    faster search.
    """
    with open(coco_json_file_path, 'r') as jf:
        coco_json = json.load(jf)
    coco_images = {image['file_name']: image for image in coco_json['images']}
    coco_annotations = defaultdict(list)
    for annotation in coco_json['annotations']:
        coco_annotations[annotation['image_id']].append(annotation)
    coco_categories = {
        category['id']: category for category in coco_json['categories']
    }
    return coco_images, coco_annotations, coco_categories


def coco_annotations2kek_objects(
        image_id: int,
        annotations: List[Dict[str, Any]],
        coco_categories: Dict[int, Dict[str, Union[int, str]]]
) -> List[KEKObject]:
    kek_objects = []
    for annotation in annotations:

        # Main object data.
        class_id = annotation['category_id']
        try:
            category = coco_categories[class_id]
        except KeyError:

            # Some smart guys used string category_id in source annotation file.
            category = coco_categories[int(class_id)]
        class_id = int(class_id) - 1
        class_name = category['name']
        kek_box = KEKBox.from_coco(annotation['bbox'])

        # Additional object data.
        # These keys should not be processed during object data dictionary
        # parsing.
        main_object_data_keys = ('category_id', 'bbox',)
        object_additional_data = cu.construct_additional_object_data(image_id)
        for key, value in annotation.items():
            if key not in main_object_data_keys:
                object_additional_data.update({key: value})

        # Category information is also information about object.
        main_object_category_keys = ('name', 'id')
        for key, value in category.items():
            if key not in main_object_category_keys:
                object_additional_data.update({key: value})
        kek_objects.append(KEKObject(class_id, class_name, kek_box,
                                     object_additional_data))
    return kek_objects


def mscoco2kek(
        image_path: str,
        base_annotation_path: str = None,
        hard: bool = True,
        coco_images: Dict[str, Dict[str, Any]] = None,
        coco_annotations: Dict[int, List[Dict[str, Any]]] = None,
        coco_categories: Dict[int, Dict[str, Union[int, str]]] = None
) -> KEKImage:
    if hard:
        return mscoco_hard2kek(
            image_path,
            coco_images,
            coco_annotations,
            coco_categories
        )
    else:
        return mscoco_simple2kek(
            image_path,
            base_annotation_path,
            coco_categories
        )


def kek2mscoco(kek_format: KEKImage, hard: bool = False) -> \
        Union[
            Tuple[
                Dict[str, Any],
                List[Dict[str, Any]],
                Dict[int, Dict[str, Union[int, str]]]
            ],
            Tuple[
                Dict[str, Any],
                Dict[int, Dict[str, Union[int, str]]]
            ]
        ]:
    if hard:
        return kek2mscoco_hard(kek_format)
    else:
        return kek2mscoco_simple(kek_format)


def kek2mscoco_simple(kek_image: KEKImage) -> \
        Tuple[
            Dict[str, Any],
            Dict[int, Dict[str, Union[int, str]]]
        ]:
    json_file = dict.fromkeys(('annotation', 'image'))
    categories = {}

    # Image data.
    image_dict = kek_image.additional_data
    image_dict.update({'file_name': kek_image.filename})
    width, height, _ = kek_image.shape
    image_dict.update({'width': width, 'height': height})
    image_dict.update({'id': kek_image.id_})
    json_file['image'] = image_dict

    # Objects data.
    annotations = []
    for kek_object in kek_image.kek_objects:
        category_id = kek_object.class_id + 1
        ms_coco_metadata = kek_object.additional_data
        ms_coco_metadata.update({'category_id': category_id,
                                 'bbox': kek_object.kek_box.to_coco_box()})
        annotations.append(ms_coco_metadata)
        category_dict = {
            'category_id': category_id,
            'name': kek_object.class_name
        }
        try:
            supercategory = kek_object.additional_data['supercategory']
            category_dict.update({'supercategory': supercategory})
        except KeyError:
            pass
        categories.update({category_id: category_dict})
    json_file['annotation'] = annotations
    return json_file, categories


def mscoco_simple2kek(
        image_path: str,
        base_annotation_path: str,
        coco_categories: Dict[int, Dict[str, Union[int, str]]]
) -> KEKImage:
    image_name = os.path.split(image_path)[-1]
    json_path = cu.construct_annotation_file_path(
        image_path,
        cu.get_target_annotation_file_extension('mscoco'),
        base_annotation_path
    )
    with open(json_path, 'r') as jf:
        labels = json.load(jf)

    # Main image data.
    image_dict = labels['image']
    image_id = image_dict['id']
    try:
        filename = image_dict['file_name']
    except KeyError:
        annotation_filename = os.path.splitext(image_name)[0] + '.json'
        cu.warn_filename_not_found(annotation_filename)
        filename = image_name
    try:
        width = image_dict['width']
        height = image_dict['height']
        depth = cu.get_image_shape(image_path)[-1]
        image_shape = (width, height, depth)
    except KeyError:
        image_shape = cu.get_image_shape(image_path)

    # Additional image data.
    # This keys should not be processed for additional image data.
    main_image_data_keys = ('id', 'file_name', 'width', 'height')
    image_additional_data = cu.construct_additional_image_data(image_path)
    for key, value in image_dict.items():
        if key not in main_image_data_keys:
            image_additional_data.update({key: value})
    annotations = labels['annotation']
    kek_objects = coco_annotations2kek_objects(
        image_id,
        annotations,
        coco_categories
    )
    return KEKImage(
        image_id,
        filename,
        image_shape,
        kek_objects,
        image_additional_data
    )


def kek2mscoco_hard(kek_format: KEKImage) -> \
        Tuple[
            Dict[str, Any],
            List[Dict[str, Any]],
            Dict[int, Dict[str, Union[int, str]]]
        ]:

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
        category_id = kek_object.class_id + 1
        ms_coco_metadata = kek_object.additional_data
        ms_coco_metadata.update({
            'category_id': category_id,
            'bbox': kek_object.kek_box.to_coco_box()
        })
        annotations.append(ms_coco_metadata)

        # Categories.
        category_dict = {'id': category_id, 'name': kek_object.class_name}
        try:
            supercategory = kek_object.additional_data['supercategory']
            category_dict.update({'supercategory': supercategory})
        except KeyError:
            pass
        categories.update({category_id: category_dict})
    return image_dict, annotations, categories


def mscoco_hard2kek(
        image_path: str,
        coco_images: Dict[str, Dict[str, Any]],
        coco_annotations: Dict[int, List[Dict[str, Any]]],
        coco_categories: Dict[int, Dict[str, Union[int, str]]]
) -> KEKImage:
    image_name = os.path.split(image_path)[-1]
    image_dict = coco_images[image_name]

    # Main image data.
    image_id = image_dict['id']
    try:
        filename = image_dict['file_name']
    except KeyError:
        cu.warn_filename_not_found('')
        filename = image_name
    try:
        width = image_dict['width']
        height = image_dict['height']
        depth = cu.get_image_shape(image_path)[-1]
        image_shape = (width, height, depth)
    except KeyError:
        image_shape = cu.get_image_shape(image_path)

    # Additional image data.
    # This keys should not be processed for additional image data.
    main_image_data_keys = ('id', 'file_name', 'width', 'height')
    image_additional_data = cu.construct_additional_image_data(image_path)
    for key, value in image_dict.items():
        if key not in main_image_data_keys:
            image_additional_data.update({key: value})
    annotations = coco_annotations[image_id]
    kek_objects = coco_annotations2kek_objects(
        image_id,
        annotations,
        coco_categories
    )
    return KEKImage(
        image_id,
        filename,
        image_shape,
        kek_objects,
        image_additional_data
    )


def save_annotation(path: str, annotation: Dict[str, Any]) -> None:
    with open(path, 'w') as jf:
        json.dump(annotation, jf)


def save_categories(path: str, categories: List[Dict[str: Any]]) -> None:
    unique_categories = {}
    for result_dict in categories:
        categories_dict = result_dict['mscoco_simple_categories']
        for category_id, category in categories_dict.items():
            unique_categories.update({category_id: category})
    category_list = [category for category in unique_categories.values()]
    with open(os.path.join(path), 'w') as jf:
        json.dump(category_list, jf)


def create_mscoco_big_dict(
        results,
        mscoco_info_path=None,
        mscoco_licenses_path=None
):
    mscoco_main_dict = {}
    if mscoco_info_path:
        add_info_section(mscoco_info_path, mscoco_main_dict)
    if mscoco_licenses_path:
        add_licenses_section(mscoco_licenses_path, mscoco_main_dict)
    mscoco_main_dict['images'] = []
    mscoco_main_dict['annotations'] = []
    mscoco_main_dict['categories'] = []
    unique_categories = {}
    for result_dict in results:
        mscoco_main_dict_piece = result_dict['mscoco_main_dict']
        mscoco_main_dict['images'].extend(mscoco_main_dict_piece['images'])
        mscoco_main_dict['annotations'].extend(
            mscoco_main_dict_piece['annotations']
        )
        categories_piece = mscoco_main_dict_piece['categories']
        for category in categories_piece:
            unique_categories.update({category['id']: category})
    category_list = [category for category in unique_categories.values()]
    mscoco_main_dict['categories'].extend(category_list)
    return mscoco_main_dict


def add_info_section(path_to_info_section, dict_to_add):
    with open(path_to_info_section, 'r') as jf:
        info = json.load(jf)
    dict_to_add['info'] = info


def add_licenses_section(path_to_licenses_section, dict_to_add):
    with open(path_to_licenses_section, 'r') as jf:
        licenses = json.load(jf)
    dict_to_add['licenses'] = licenses['licenses']
