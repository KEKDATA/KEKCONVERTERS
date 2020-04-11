import os
import json
import multiprocessing as mp
from functools import partial

from conversion.converters import mscoco as mc
from conversion.converters import darknet as dn
from conversion.converters import pascalvoc as pv

from conversion.script_utils import parse_args
from conversion.script_utils import image_iter
from conversion.script_utils import get_converters
from conversion.script_utils import get_class_mapper
from conversion.script_utils import get_source_mscoco_annotations
from conversion.script_utils import get_target_mscoco_dictionaries
from conversion.script_utils import get_target_annotation_file_extension

if __name__ == '__main__':
    args = parse_args()
    from_converter, to_converter = get_converters(
        args.src_annotation,
        args.dst_annotation
    )
    class_mapper = None
    if args.class_mapper_path:
        class_mapper = get_class_mapper(args.class_mapper_path)
    coco_images, coco_annotations, coco_categories = None, None, None
    if args.src_annotation == 'mscoco':
        (coco_images, coco_annotations,
         coco_categories) = get_source_mscoco_annotations(
            args.ano_path,
            args.mscoco_hard,
            args.mscoco_categories_path
        )
    mscoco_big_dict = None
    categories = None
    if args.dst_annotation == 'mscoco':
        mscoco_big_dict, categories = get_target_mscoco_dictionaries(
            args.mscoco_hard,
            args.mscoco_info_path,
            args.mscoco_licenses_path
        )
        # if args.mscoco_hard:
        #     mscoco_big_dict = {}
        #     if args.mscoco_info_path:
        #         with open(args.mscoco_info_path, 'r') as jf:
        #             mscoco_info_section = json.load(jf)
        #         mscoco_big_dict['info'] = mscoco_info_section
        #     if args.mscoco_licenses_path:
        #         with open(args.mscoco_licenses_path, 'r') as jf:
        #             mscoco_licenses_section = json.load(jf)
        #         mscoco_big_dict['licenses'] = mscoco_licenses_section
        #     mscoco_big_dict['images'] = []
        #     mscoco_big_dict['annotations'] = []
        #     mscoco_big_dict['categories'] = {}
        # else:
        #     categories = {}
    from_args = {
        'darknet': (class_mapper, args.ano_path if args.ano_path else None),
        'pascalvoc': (class_mapper, args.ano_path if args.ano_path else None),
        'mscoco': (args.ano_path if args.ano_path else None, args.mscoco_hard,
                   coco_images, coco_annotations, coco_categories)
    }.get(args.src_annotation)

    img_exts = tuple(args.img_exts.split(','))
    save_annotation_file_extension = get_target_annotation_file_extension(
        args.dst_annotation
    )
    # save_annotation_file_extension = {
    #     'darknet': '.txt',
    #     'pascalvoc': '.xml',
    #     'mscoco': '.json'
    # }.get(args.dst_annotation)
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
