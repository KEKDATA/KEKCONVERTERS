"""This module contains converter's classes."""
import os
from typing import Union, Dict

from PIL import Image

from .kek_format import KEKBox, KEKObject, KEKFormat


class BaseConverter:
    def __init__(self, annotation: str) -> None:
        self.annotation = annotation

    def raise_not_implemented(self, *args, **kwargs):
        raise NotImplementedError('Conversion for {} currently is not supported. '
                                  'Please feel free to create corresponding issue.'.format(self.annotation))


class ToKEKFormatConverter(BaseConverter):
    def __init__(self, annotation: str, class_mapper: Dict[Union[str, int], Union[int, str]]) -> None:
        """
        :param annotation: String represented annotation type. 'mscoco', 'pascalvoc', 'darknet', etc;
        :param class_mapper: Dictionary contained mapping for class names or class ids. For example:
                             for Darknet format: {0: 'person', 1: 'car', ...}
                             for PASCAL VOC format: {'person': 0, 'car': 1, ...}
                             etc.
        """
        super().__init__(annotation)
        self.class_mapper = class_mapper

    def _from_darknet(self, image: os.DirEntry, base_annotation_path: str = None) -> KEKFormat:
        """
        Converts Darknet annotation format for given image to KEKFormat representation.

        :param image: Source image;
        :param base_annotation_path: Path to directory which contains .txt annotation
                                     files.

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
                KEKObject(class_name=self.class_mapper[int(class_id)], class_id=int(class_id),
                          kek_box=kek_box)
            )
        return KEKFormat(kek_objects, image_metadata)

    def _from_pascal_voc(self) -> KEKFormat:
        pass

    def _from_ms_coco(self) -> KEKFormat:
        pass

    def convert(self, *args, **kwargs) -> KEKFormat:
        return {
            'darknet': self._from_darknet,
            'pascalvoc': self._from_pascal_voc,
            'mscoco': self._from_ms_coco
        }.get(self.annotation, self.raise_not_implemented)(*args, **kwargs)


class FromKEKFormatCOnverter(BaseConverter):
    def _to_darknet(self):
        pass

    def _to_pascal_voc(self):
        pass

    def _to_ms_coco(self):
        pass

    def convert(self, *args, **kwargs):
        return {
            'darknet': self._to_darknet,
            'pascalvoc': self._to_pascal_voc,
            'mscoco': self._to_ms_coco
        }.get(self.annotation, self.raise_not_implemented)(*args, **kwargs)