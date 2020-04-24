"""Home for helpers functions for console scripts."""
import os
import json
import argparse as ap
from functools import partial
from typing import Dict, Any, Tuple, Callable, DefaultDict, List, Union, Iterable

import yaml

import conversion.converters.converters_utils as cu
from conversion.converters import mscoco as mc
from conversion.converters import darknet as dn
from conversion.converters import pascalvoc as pv


def filter_by_extensions(filename, image_extensions) -> bool:
    return os.path.splitext(filename)[-1] in image_extensions


def parse_args() -> ap.Namespace:
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--path_to_yaml_config',
        '-config',
        type=str,
        required=True,
        help="Path to your .yaml configuration file"
    )
    return parser.parse_args()


def parse_config_file(path_to_config_file: str) -> Dict[str, Any]:
    with open(path_to_config_file, 'r') as yf:
        config_dict = yaml.load(yf, Loader=yaml.SafeLoader)
    return config_dict


def get_converters(source_annotation_name: str, target_annotation_name: str) \
        -> Tuple[Callable, Callable]:
    from_converter = {
        'darknet': dn.darknet2kek,
        'pascalvoc': pv.pascalvoc2kek,
        'mscoco': mc.mscoco2kek
    }.get(source_annotation_name)
    to_converter = {
        'darknet': dn.kek2darknet,
        'pascalvoc': pv.kek2pascalvoc,
        'mscoco': mc.kek2mscoco
    }.get(target_annotation_name)
    return from_converter, to_converter


def get_saver(target_annotation_name: str) -> Callable:
    return {
        'darknet': dn.save_annotation,
        'pascalvoc': pv.save_annotation,
        'mscoco': pv.save_annotation
    }.get(target_annotation_name)


def get_class_mapper(class_mapper_path: str) -> Dict[str, Any]:
    with open(class_mapper_path, 'r') as jf:
        return json.load(jf)


def get_source_mscoco_annotations(
        annotation_path: str,
        hard: bool,
        mscoco_categories_path: str
) -> Union[
        Tuple[
            Dict[str, Dict[str, Any]],
            DefaultDict[int, List[Dict[str, Any]]],
            Dict[int, Dict[str, Any]]
        ],
        Tuple[
            None,
            None,
            Dict[int, Dict[str, Any]]
        ]
     ]:
    if hard:
        return mc.construct_mscoco_dicts(annotation_path)
    else:
        with open(mscoco_categories_path, 'r') as jf:
            coco_categories = json.load(jf)
        return (
            None,
            None,
            {category['id']: category for category in coco_categories}
        )


def conversion_loop(
        image_paths: Iterable[str],
        save_annotation_path: str,
        source_annotation_name: str,
        target_annotation_name: str,
        from_converter_function: Callable,
        from_converter_function_args: tuple,
        to_converter_function: Callable,
        save_function: Callable,
        target_annotation_file_extension: str,
        mscoco_hard: bool = False
) -> Dict[
        str,
        Union[
              None,
              Dict[str, Any],
              Dict[
                  str,
                  Union[
                      List[Dict[str, Any]],
                      Dict[str, Any]
                  ]
              ]
        ]
     ]:
    """GOD BLESS OUR CONVERSION, GUYS"""
    categories = {}
    coco_simple_categories = None
    if target_annotation_name == 'mscoco' and mscoco_hard:
        mscoco_main_dict = {
            'images': [],
            'annotations': [],
            'categories': {}
        }
    else:
        mscoco_main_dict = None
    result_dict = {'mscoco_simple_categories': None, 'mscoco_main_dict': None}
    for image_id, image_path in enumerate(image_paths):
        if source_annotation_name == 'mscoco':
            kek_format = from_converter_function(
                image_path,
                *from_converter_function_args
            )
        else:
            kek_format = from_converter_function(
                image_path,
                image_id,
                *from_converter_function_args
            )
        if target_annotation_name == 'mscoco' and mscoco_hard:
            to_converter_function_args = (kek_format, mscoco_hard)
        else:
            to_converter_function_args = (kek_format, )
        target_format = to_converter_function(*to_converter_function_args)
        if len(target_format) == 2:
            target_format, coco_simple_categories = target_format
        if not mscoco_main_dict:

            # Target format consists of multiple annotation files.
            annotation_file_path = cu.construct_annotation_file_path(
                image_path,
                target_annotation_file_extension,
                save_annotation_path
            )
            save_function(annotation_file_path, target_format)
            # with open(annotation_file_path, 'w') as af:
            #     writer_function = {
            #         'darknet': af.writelines,
            #         'pascalvoc': af.write,
            #         'mscoco': partial(json.dump, fp=af)
            #     }.get(target_annotation_name)
            #     writer_function(target_format)
            if coco_simple_categories:
                for category_id, category in coco_simple_categories.items():
                    categories.update({category_id: category})
        else:

            # Target format is single HUGE BIG DICT, GUYS.
            image_dict, annotations, categories = target_format
            mscoco_main_dict['images'].append(image_dict)
            mscoco_main_dict['annotations'].extend(annotations)
            for category_id, category in categories.items():
                mscoco_main_dict['categories'].update({category_id: category})
    if coco_simple_categories:
        result_dict['mscoco_simple_categories'] = categories
    if mscoco_main_dict:
        category_list = []
        for category_id, category in mscoco_main_dict['categories'].items():
            category_list.append(category)
        mscoco_main_dict['categories'] = category_list
        result_dict['mscoco_main_dict'] = mscoco_main_dict
    return result_dict


def get_chunks(image_paths: List[str], n_jobs: int) -> List[List[str]]:
    div, mod = divmod(len(image_paths), n_jobs)
    return [image_paths[i * div + min(i, mod):(i + 1) * div + min(i + 1, mod)]
            for i in range(n_jobs)]


def get_full_paths(path_to_images: str, image_extensions: Iterable[str]) -> \
        List[str]:
    full_image_paths = []
    is_image = partial(filter_by_extensions, image_extensions=image_extensions)
    for image_name in filter(is_image, os.listdir(path_to_images)):
        full_image_paths.append(os.path.join(path_to_images, image_name))
    return full_image_paths


def process_conversion_results(
        results: List[
                        Dict[
                            str,
                            Union[
                                None,
                                Dict[str, Any],
                                Dict[
                                    str,
                                    Union[
                                        List[Dict[str, Any]],
                                        Dict[str, Any]
                                    ]
                                ]
                            ]
                        ]
                 ],
        save_path: str,
        mscoco_licenses_path: str = None,
        mscoco_info_path: str = None
) -> None:
    if results[0]['mscoco_main_dict']:
        mscoco_main_dict = mc.create_mscoco_big_dict(
            results,
            mscoco_info_path,
            mscoco_licenses_path
        )
        mc.save_annotation(save_path, mscoco_main_dict)
    elif results[0]['mscoco_simple_categories']:
        mc.save_categories(save_path, results)
