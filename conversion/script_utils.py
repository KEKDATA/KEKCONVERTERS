"""Home for helpers functions for console scripts."""
import os
import json
import argparse as ap
from typing import Iterable

from conversion.converters import mscoco as mc
from conversion.converters import darknet as dn
from conversion.converters import pascalvoc as pv


SRC_ANNOTATION_HELP = 'Source annotation type. One of \'darknet\', \'pascalvoc\', \'mscoco\'.'
DST_ANNOTATION_HELP = 'Target annotation type. One of \'darknet\', \'pascalvoc\', \'mscoco\'.'
IMAGE_PATH_HELP = 'Path to your images.'
IMAGE_EXTS_HELP = 'Image extensions for your dataset separated by \',\'. For example \'.img,.jpeg,.jpg,.bmp\''
ANNOTATION_PATH_HELP = 'Path to your source annotation files.'
SAVE_PATH_HELP = 'Path to directory for new converted annotation files.'

MSCOCO_HARD_HELP = """True if your source dataset annotation file is single file like source MS 
COCO trainval2017."""
MSCOCO_INFO_PATH_HELP = """If you want your hard MSCOCO single annotation file to contain the
'info' section then provide path to your info.json to this argument."""
MSCOCO_LICENSES_PATH_HELP = """If you want your hard MSCOCO single annotation file to contain the
'licenses' section then provide path to your licenses.json to this argument."""
MSCOCO_CATEGORIES_PATH_HELP = """If you use simple MSCOCO mode (when single image has single 
annotation .json file) then you should provide path to MSCOCO categories.json file to this
argument."""

DARKNET_MAPPING_PATH_HELP = 'Path to mapper for integer class labels to string class labels for Darknet.'
PASCALVOC_MAPPING_PATH_HELP = 'Path to mapper for string class labels to integer class labels for PASCAL VOC.'

NJOBS_HELP = 'Amount of processes to use for conversion.'


def image_iter(path: str, image_exts: Iterable[str]) -> filter:
    return filter(lambda filename: os.path.splitext(filename)[-1] in image_exts, os.scandir(path))


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--src_annotation', '-sa', type=str, required=True, help=SRC_ANNOTATION_HELP)
    parser.add_argument('--dst_annotation', '-da', type=str, required=True, help=DST_ANNOTATION_HELP)
    parser.add_argument('--img_path', '-ip', type=str, required=True, help=IMAGE_PATH_HELP)
    parser.add_argument('--img_exts', '-ie', type=str, required=True, help=IMAGE_EXTS_HELP)
    parser.add_argument('--ano_path', '-ap', type=str, required=False, help=ANNOTATION_PATH_HELP)
    parser.add_argument('--save_path', '-sp', type=str, required=True, help=SAVE_PATH_HELP)
    parser.add_argument('--mscoco_hard', '-msh', type=bool, required=False, nargs='?', const=True, default=False,
                        help=MSCOCO_HARD_HELP)
    parser.add_argument('--mscoco_info_path', '-mip', type=str, required=False, help=MSCOCO_INFO_PATH_HELP)
    parser.add_argument('--mscoco_licenses_path', '-mlp', type=str, required=False, help=MSCOCO_LICENSES_PATH_HELP)
    parser.add_argument('--mscoco_categories_path', '-mcp', type=str, required=False, help=MSCOCO_CATEGORIES_PATH_HELP)
    parser.add_argument('--class_mapper_path', '-cmp', type=str,
                        required=False, help='Path to your mapping')
    parser.add_argument('--n_jobs', '-j', type=int, required=False, default=None, help=NJOBS_HELP)
    return parser.parse_args()


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


def get_target_mscoco_dictionaries(hard: bool, mscoco_info_path: str = None,
                                   mscoco_licenses_path: str = None):

    if hard:
        mscoco_big_dict = {}
        if mscoco_info_path:
            with open(mscoco_info_path, 'r') as jf:
                mscoco_info_section = json.load(jf)
            mscoco_big_dict['info'] = mscoco_info_section
        if mscoco_licenses_path:
            with open(mscoco_licenses_path, 'r') as jf:
                mscoco_licenses_section = json.load(jf)
            mscoco_big_dict['licenses'] = mscoco_licenses_section
        mscoco_big_dict['images'] = []
        mscoco_big_dict['annotations'] = []
        mscoco_big_dict['categories'] = {}
        return mscoco_big_dict, None
    else:
        categories = {}
        return None, categories


def get_target_annotation_file_extension(target_annotation_name: str):
    return {
        'darknet': '.txt',
        'pascalvoc': '.xml',
        'mscoco': '.json'
    }.get(target_annotation_name)
