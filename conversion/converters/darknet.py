import os
from typing import Dict, List

from PIL import Image

from conversion.entities import KEKBox, KEKObject, KEKFormat


def darknet2kek(image: os.DirEntry, class_mapper: Dict[int, str],
                base_annotation_path: str = None) -> KEKFormat:
    """
    Converts Darknet annotation format for given image to KEKFormat
    representation.

    :param image: Source image;
    :param class_mapper: Mapping for integer labels to string labels.
                         For example, {0: 'car', 1: 'person', ...};
    :param base_annotation_path: Path to directory which contains .txt annotation files.

    :return: KEKFormat representation.
    """
    image_name, image_ext = os.path.splitext(image.name)
    if not base_annotation_path:
        base_txt_path = os.path.split(image.path)[0]
    else:
        base_txt_path = base_annotation_path
    txt_name = '.'.join([image_name, 'txt'])
    txt_path = os.path.join(base_txt_path, txt_name)
    with open(txt_path, 'r') as label_txt:
        darknet_labels = label_txt.readlines()
    pil_image = Image.open(image.path)
    image_width, image_height = pil_image.size
    image_depth = len(pil_image.getbands())
    image_metadata = {'image_width': image_width, 'image_height': image_height,
                      'image_depth': image_depth}
    kek_objects = []
    for darknet_label in darknet_labels:
        first_space = darknet_label.find(' ')
        class_id = darknet_label[:first_space]
        box = darknet_label[first_space + 1:]
        kek_box = KEKBox.from_darknet(box, (image_height, image_width, image_depth))
        kek_objects.append(
            KEKObject(class_name=class_mapper[int(class_id)], class_id=int(class_id),
                      kek_box=kek_box)
        )
    return KEKFormat(kek_objects, image_metadata)


def kek2darknet(kek_format: KEKFormat) -> List[str]:
    darknet_labels = []
    image_width = kek_format.image_metadata['image_width']
    image_height = kek_format.image_metadata['image_height']
    image_depth = kek_format.image_metadata['image_depth']
    image_shape = (image_height, image_width, image_depth)
    for kek_object in kek_format.kek_objects:
        class_id = str(kek_object.class_id)
        darknet_box = kek_object.kek_box.to_darknet_box(image_shape)
        darknet_labels.append(
            ' '.join([class_id, *map(str, darknet_box)]) + '\n'
        )
    return darknet_labels
