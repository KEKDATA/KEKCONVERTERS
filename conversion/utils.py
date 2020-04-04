import os
import warnings

from PIL import Image


def warn_filename_not_found(annotation_file_name: str):
    """We should warn user if annotations doesn't contain filename in annotation
    file. We can get this filename from corresponding annotation file. And if
    this we should warn users about it."""
    warnings.warn('Annotation file {} has no filename tag.\nGetting filename '
                  'from file...'.format(annotation_file_name))


def construct_annotation_file_path(image: os.DirEntry, annotation_file_extension: str,
                                   base_annotation_path: str = None) -> str:
    image_name, image_ext = os.path.splitext(image.name)
    if not base_annotation_path:
        base_file_path = os.path.split(image.path)[0]
    else:
        base_file_path = base_annotation_path
    annotation_filename = '.'.join([image_name, annotation_file_extension])
    return os.path.join(base_file_path, annotation_filename)


def get_image_shape(image: os.DirEntry):
    pil_image = Image.open(image.path)
    width, height = pil_image.size
    depth = len(pil_image.getbands())
    return width, height, depth


def construct_additional_image_data(image: os.DirEntry):
    path = image.path
    folder = os.path.split(os.path.split(image.path)[0])[-1]
    image_additional_data = {'path': path, 'folder': folder}
    return image_additional_data


def construct_additional_object_data(image_id: int):
    return {'image_id': image_id}
