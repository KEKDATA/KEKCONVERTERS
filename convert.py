import os
import json
import argparse as ap
from typing import Iterable
import multiprocessing as mp
from functools import partial

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


if __name__ == '__main__':
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
    parser.add_argument('--darknet_mapping_path', '-dm', type=str, required=False, help=DARKNET_MAPPING_PATH_HELP)
    parser.add_argument('--pascalvoc_mapping_path', '-pvm', type=str, required=False, help=PASCALVOC_MAPPING_PATH_HELP)
    parser.add_argument('--n_jobs', '-j', type=int, required=False, default=None, help=NJOBS_HELP)
    args = parser.parse_args()

    from_converter = {'darknet': dn.darknet2kek,
                      'pascalvoc': pv.pascalvoc2kek,
                      'mscoco': mc.mscoco2kek}.get(args.src_annotation)
    to_converter = {'darknet': dn.kek2darknet,
                    'pascalvoc': pv.kek2pascalvoc,
                    'mscoco': mc.kek2mscoco}.get(args.dst_annotation)

    darknet_mapper = None
    pascalvoc_mapper = None
    if args.darknet_mapping_path:
        with open(args.darknet_mapping_path, 'r') as jf:
            darknet_mapper = json.load(jf)
        darknet_mapper = {int(key): value for key, value in darknet_mapper.items()}
    elif args.pascalvoc_mapping_path:
        with open(args.pascalvoc_mapping_path, 'r') as jf:
            pascalvoc_mapper = json.load(jf)
    coco_images, coco_annotations, coco_categories = None, None, None
    if args.src_annotation == 'mscoco':
        if args.mscoco_hard:
            coco_images, coco_annotations, coco_categories = mc.construct_mscoco_dicts(args.ano_path)
        else:
            coco_images = None
            coco_annotations = None
            with open(args.mscoco_categories_path, 'r') as jf:
                coco_categories = json.load(jf)
                coco_categories = {
                    category['id']: category for category in coco_categories}

    mscoco_big_dict = None
    categories = None
    if args.dst_annotation == 'mscoco':
        if args.mscoco_hard:
            mscoco_big_dict = {}
            if args.mscoco_info_path:
                with open(args.mscoco_info_path, 'r') as jf:
                    mscoco_info_section = json.load(jf)
                mscoco_big_dict['info'] = mscoco_info_section
            if args.mscoco_licenses_path:
                with open(args.mscoco_licenses_path, 'r') as jf:
                    mscoco_licenses_section = json.load(jf)
                mscoco_big_dict['licenses'] = mscoco_licenses_section
            mscoco_big_dict['images'] = []
            mscoco_big_dict['annotations'] = []
            mscoco_big_dict['categories'] = {}
        else:
            categories = {}

    from_args = {
        'darknet': (darknet_mapper, args.ano_path if args.ano_path else None),
        'pascalvoc': (pascalvoc_mapper, args.ano_path if args.ano_path else None),
        'mscoco': (args.ano_path if args.ano_path else None, args.mscoco_hard,
                   coco_images, coco_annotations, coco_categories)
    }.get(args.src_annotation)

    img_exts = tuple(args.img_exts.split(','))
    save_annotation_file_extension = {
        'darknet': '.txt',
        'pascalvoc': '.xml',
        'mscoco': '.json'
    }.get(args.dst_annotation)
    cats = None
    for image_id, image in enumerate(image_iter(args.img_path, img_exts)):
        if args.src_annotation == 'mscoco':
            kek_format = from_converter(image, *from_args)
        else:
            kek_format = from_converter(image, image_id, *from_args)
        if args.dst_annotation == 'mscoco' and args.mscoco_hard:
            to_args = (kek_format, args.mscoco_hard)
        else:
            to_args = (kek_format, )
        target_format = to_converter(*to_args)
        if len(target_format) == 2:
            target_format, cats = target_format

        if mscoco_big_dict is None:
            image_name, image_ext = os.path.splitext(image.name)
            annotation_file_name = ''.join([image_name, save_annotation_file_extension])
            annotation_file_path = os.path.join(args.save_path, annotation_file_name)
            with open(annotation_file_path, 'w') as af:
                writer_func = {
                    'darknet': af.writelines,
                    'pascalvoc': af.write,
                    'mscoco': partial(json.dump, fp=af)
                }.get(args.dst_annotation)
                writer_func(target_format)
                if cats:
                    for category_id, category in cats.items():
                        categories.update({category_id: category})
        else:
            image_dict, annotations, categories = target_format
            mscoco_big_dict['images'].append(image_dict)
            mscoco_big_dict['annotations'].extend(annotations)
            for category_id, category in categories.items():
                mscoco_big_dict['categories'].update({category_id: category})

    if cats:
        category_list = []
        for category_id, category in categories.items():
            category_list.append(category)
        with open(os.path.join(args.save_path, 'categories.json'), 'w') as jf:
            json.dump(category_list, jf)
    if mscoco_big_dict:
        category_list = []
        for category_id, category in mscoco_big_dict['categories'].items():
            category_list.append(category)
        mscoco_big_dict['categories'] = category_list
        with open(os.path.join(args.save_path, 'annotation.json'), 'w') as jf:
            json.dump(mscoco_big_dict, jf)
