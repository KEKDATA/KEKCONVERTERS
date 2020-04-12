"""Home for helpers functions for console scripts."""
import os
import json
import argparse as ap
from functools import partial
from typing import Iterable

import yaml

import conversion.converters.converters_utils as cu
from conversion.converters import mscoco as mc
from conversion.converters import darknet as dn
from conversion.converters import pascalvoc as pv


def image_iter(path: str, image_exts: Iterable[str]) -> filter:
    return filter(lambda filename: os.path.splitext(filename)[-1] in image_exts, os.scandir(path))


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--path_to_yaml_config',
        '-config',
        type=str,
        required=True,
        help="Path to your .yaml configuration file"
    )
    return parser.parse_args()


def parse_config_file(path_to_config_file):
    with open(path_to_config_file, 'r') as yf:
        config_dict = yaml.load(yf, Loader=yaml.SafeLoader)
    return config_dict


def get_converters(source_annotation_name: str, target_annotation_name:str):
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


def get_class_mapper(class_mapper_path: str) -> dict:
    with open(class_mapper_path, 'r') as jf:
        return json.load(jf)


def get_source_mscoco_annotations(annotation_path: str, hard: bool,
                                  mscoco_categories_path: str):
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


def get_target_annotation_file_extension(target_annotation_name: str):
    return {
        'darknet': '.txt',
        'pascalvoc': '.xml',
        'mscoco': '.json'
    }.get(target_annotation_name)


def conversion_loop(
        image_paths,
        save_annotation_path,
        source_annotation_name,
        target_annotation_name,
        from_converter_function,
        from_converter_function_args,
        to_converter_function,
        target_annotation_file_extension,
        mscoco_hard=False
):
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
            with open(annotation_file_path, 'w') as af:
                writer_function = {
                    'darknet': af.writelines,
                    'pascalvoc': af.write,
                    'mscoco': partial(json.dump, fp=af)
                }.get(target_annotation_name)
                writer_function(target_format)
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


def get_chunks(image_paths, n_jobs):
    div, mod = divmod(len(image_paths), n_jobs)
    return [image_paths[i * div + min(i, mod):(i + 1) * div + min(i + 1, mod)]
            for i in range(n_jobs)]


def get_full_paths(image_names, base_path):
    return [os.path.join(base_path, image_name) for image_name in image_names]


def process_conversion_results(results, save_path, mscoco_licenses_path=None,
                               mscoco_info_path=None):
    if results[0]['mscoco_main_dict']:
        # Due to the chosen parallelization strategy, we know for sure that
        # inside the large MS COCO dictionary, the lists 'images' and
        # 'annotations' are unique in terms of processes. However, we cannot be
        # sure that the list with the categories is unique, because different
        # images can have different objects that belong to the same category.
        # Therefore, categories need to be filtered.
        mscoco_main_dict = {}
        if mscoco_info_path:
            with open(mscoco_info_path, 'r') as jf:
                info = json.load(jf)
            mscoco_main_dict['info'] = info
        if mscoco_licenses_path:
            with open(mscoco_licenses_path, 'r') as jf:
                licenses = json.load(jf)
            mscoco_main_dict['licenses'] = licenses['licenses']
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
        with open(os.path.join(save_path), 'w') as jf:
            json.dump(mscoco_main_dict, jf)
    elif results[0]['mscoco_simple_categories']:
        # Different images might have objects with same categories. So result
        # categories dictionary might contain same category entities and we
        # need to filter it.
        unique_categories = {}
        for result_dict in results:
            categories_dict = result_dict['mscoco_simple_categories']
            for category_id, category in categories_dict.items():
                unique_categories.update({category_id: category})
        category_list = [category for category in unique_categories.values()]
        with open(os.path.join(save_path, 'categories.json'), 'w') as jf:
            json.dump(category_list, jf)
