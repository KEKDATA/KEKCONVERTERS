import multiprocessing as mp
from functools import partial

import conversion.converters.converters_utils as cu

from conversion.script_utils import parse_args
from conversion.script_utils import get_chunks
from conversion.script_utils import get_full_paths
from conversion.script_utils import get_converters
from conversion.script_utils import conversion_loop
from conversion.script_utils import get_class_mapper
from conversion.script_utils import parse_config_file
from conversion.script_utils import process_conversion_results
from conversion.script_utils import get_source_mscoco_annotations


if __name__ == '__main__':
    args = parse_args()
    config_dict = parse_config_file(args.path_to_yaml_config)
    from_converter, to_converter = get_converters(
        config_dict['source_annotation_name'],
        config_dict['target_annotation_name']
    )
    class_mapper = None
    if config_dict['class_mapper_path']:
        class_mapper = get_class_mapper(config_dict['class_mapper_path'])
    mscoco_stuff = (None, None, None)
    if config_dict['source_annotation_name'] == 'mscoco':

        # coco_images, coco_annotations, coco_categories.
        mscoco_stuff = get_source_mscoco_annotations(
            config_dict['path_to_annotations'],
            config_dict['mscoco_hard_mode'],
            config_dict['mscoco_simple_categories_path']
        )
    from_args = {
        'darknet': (class_mapper, config_dict['path_to_annotations']),
        'pascalvoc': (class_mapper, config_dict['path_to_annotations']),
        'mscoco': (
            config_dict['path_to_annotations'],
            config_dict['mscoco_hard_mode'],
            *mscoco_stuff
        )
    }.get(config_dict['target_annotation_name'])
    save_annotation_file_extension = cu.get_target_annotation_file_extension(
        config_dict['target_annotation_name']
    )
    image_paths = get_full_paths(
        config_dict['path_to_images'],
        config_dict['image_extensions']
    )
    data_chunks = get_chunks(image_paths, config_dict['n_jobs'])
    partial_conversion_loop = partial(
        conversion_loop,
        save_annotation_path=config_dict['save_path'],
        source_annotation_name=config_dict['source_annotation_name'],
        target_annotation_name=config_dict['target_annotation_name'],
        from_converter_function=from_converter,
        from_converter_function_args=from_args,
        to_converter_function=to_converter,
        target_annotation_file_extension=save_annotation_file_extension,
        mscoco_hard=config_dict['mscoco_hard_mode']
    )
    with mp.Pool(processes=config_dict['n_jobs']) as pool:
        results = pool.map(partial_conversion_loop, data_chunks)
    process_conversion_results(
        results,
        config_dict['save_path'],
        config_dict['mscoco_licenses_section_path'],
        config_dict['mscoco_info_section_path']
    )
