import os
from typing import Dict, List

from conversion.converters import converters_utils as cu
from conversion.entities import KEKBox, KEKObject, KEKImage


def darknet2kek(image_path: str, image_id: int, class_mapper: Dict[str, str],
                base_annotation_path: str = None) -> KEKImage:

    # Necessary image data.
    filename = os.path.split(image_path)[-1]
    image_shape = cu.get_image_shape(image_path)

    # Additional image data.
    image_additional_data = cu.construct_additional_image_data(image_path)

    # Object data.
    txt_path = cu.construct_annotation_file_path(
        image_path,
        cu.get_target_annotation_file_extension('darknet'),
        base_annotation_path
    )
    with open(txt_path, 'r') as label_txt:
        darknet_labels = label_txt.readlines()
    kek_objects = []
    for darknet_label in darknet_labels:
        first_space = darknet_label.find(' ')
        class_id = darknet_label[:first_space]
        box = darknet_label[first_space + 1:]
        kek_box = KEKBox.from_darknet(box, image_shape)
        object_additional_data = cu.construct_additional_object_data(image_id)
        kek_objects.append(
            KEKObject(
                class_id=int(class_id),
                kek_box=kek_box,
                class_name=class_mapper[class_id],
                object_additional_data=object_additional_data
            )
        )

    return KEKImage(
        image_id,
        filename,
        image_shape,
        kek_objects,
        image_additional_data
    )


def kek2darknet(kek_image: KEKImage) -> List[str]:
    darknet_labels = []
    for kek_object in kek_image.kek_objects:
        class_id = str(kek_object.class_id)
        darknet_box = kek_object.kek_box.to_darknet_box(kek_image.shape)
        darknet_labels.append(
            ' '.join([class_id, *map(str, darknet_box)]) + '\n'
        )
    return darknet_labels
