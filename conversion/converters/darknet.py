import os
import json
from typing import Dict, List

from PIL import Image

from conversion.utils import get_image_shape
from conversion.utils import construct_annotation_file_path
from conversion.utils import construct_additional_image_data
from conversion.utils import construct_additional_object_data
from conversion.entities import KEKBox, KEKObject, KEKImage


def darknet2kek(image: os.DirEntry, image_id: int, class_mapper: Dict[str, str],
                base_annotation_path: str = None) -> KEKImage:
    """
    Converts Darknet annotation format for given image to KEKFormat
    representation.

    :param image: Source image;
    :param image_id: Image ID, LOL;
    :param class_mapper: Mapping for integer labels to string labels.
                         For example, {0: 'car', 1: 'person', ...};
    :param base_annotation_path: Path to directory which contains .txt
                                 annotation files.

    :return: KEKFormat representation.
    """
    # Necessary image data.
    filename = image.name
    image_shape = get_image_shape(image)

    # Additional image data.
    image_additional_data = construct_additional_image_data(image)

    # Object data.
    txt_path = construct_annotation_file_path(image, 'txt',
                                              base_annotation_path)
    with open(txt_path, 'r') as label_txt:
        darknet_labels = label_txt.readlines()
    kek_objects = []
    for darknet_label in darknet_labels:
        first_space = darknet_label.find(' ')
        class_id = darknet_label[:first_space]
        box = darknet_label[first_space + 1:]
        kek_box = KEKBox.from_darknet(box, image_shape)
        # Darknet has no additional data about objects on image except image id.
        object_additional_data = construct_additional_object_data(image_id)
        kek_objects.append(
            KEKObject(class_id=int(class_id), kek_box=kek_box,
                      class_name=class_mapper[int(class_id)],
                      object_additional_data=object_additional_data))

    return KEKImage(image_id, filename, image_shape, kek_objects,
                    image_additional_data)


def kek2darknet(kek_image: KEKImage) -> List[str]:
    darknet_labels = []
    for kek_object in kek_image.kek_objects:
        class_id = str(kek_object.class_id)
        darknet_box = kek_object.kek_box.to_darknet_box(kek_image.shape)
        darknet_labels.append(' '.join([class_id,
                                        *map(str, darknet_box)]) + '\n')
    return darknet_labels
