import os
import warnings

from PIL import Image


def warn_filename_not_found(annotation_file_name: str):
    """We should warn user if annotations doesn't contain filename in annotation
    file. We can get this filename from corresponding annotation file. And if
    this we should warn users about it."""
    warnings.warn('Annotation file {} has no filename tag.\nGetting filename '
                  'from file...'.format(annotation_file_name))


def construct_annotation_file_path(image_path: str,
                                   annotation_file_extension: str,
                                   base_annotation_path: str = None) -> str:
    image_name, image_ext = os.path.splitext(os.path.split(image_path)[-1])
    if not base_annotation_path:
        base_file_path = os.path.split(image_path)[0]
    else:
        base_file_path = base_annotation_path
    annotation_filename = ''.join([image_name, annotation_file_extension])
    return os.path.join(base_file_path, annotation_filename)


def get_image_shape(image_path: str):
    pil_image = Image.open(image_path)
    width, height = pil_image.size
    depth = len(pil_image.getbands())
    return width, height, depth


def construct_additional_image_data(image_path: str):
    folder = os.path.split(os.path.split(image_path)[0])[-1]
    image_additional_data = {'path': image_path, 'folder': folder}
    return image_additional_data


def construct_additional_object_data(image_id: int):
    return {'image_id': image_id}


def get_target_annotation_file_extension(target_annotation_name: str):
    return {
        'darknet': '.txt',
        'pascalvoc': '.xml',
        'mscoco': '.json'
    }.get(target_annotation_name)
