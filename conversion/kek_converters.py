"""This module contains converter's classes."""
import os
import xml.etree.ElementTree as ET
from typing import Union, Dict, List
from xml.dom.minidom import parseString


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
                KEKObject(class_name=self.class_mapper[int(class_id)], class_id=int(class_id),
                          kek_box=kek_box)
            )
        return KEKFormat(kek_objects, image_metadata)

    def _from_pascal_voc(self, image: os.DirEntry, base_annotation_path: str = None) -> KEKFormat:
        """
        Converts PASCAL VOC annotation format for given image to KEKFormat representation.

        :param image: Source image;
        :param base_annotation_path: Path to directory which contains .xml annotation files.

        :return: KEKFormat representation.
        """
        image_name, image_ext = os.path.splitext(image.name)
        if not base_annotation_path:
            base_xml_path = os.path.split(image.path)[0]
        else:
            base_xml_path = base_annotation_path
        xml_name = '.'.join([image_name, 'xml'])
        xml_path = os.path.join(base_xml_path, xml_name)
        root = ET.parse(xml_path)
        image_metadata = {
            'folder': root.find('folder').text,
            'filename': root.find('filename').text,
            'path': root.find('path').text,
            'database': root.find('source').find('database').text,
            'image_width': int(root.find('size').find('width').text),
            'image_height': int(root.find('size').find('height').text),
            'image_depth': int(root.find('size').find('depth').text),
            'segmented': int(root.find('segmented').text)
        }
        objects = root.findall('object')
        kek_objects = []
        for object_ in objects:
            class_name = object_.find('name').text
            class_id = self.class_mapper[class_name]
            pv_metadata = {
                'pose': object_.find('pose').text,
                'truncated': object_.find('truncated').text,
                'difficult': object_.find('difficult').text
            }
            box = [int(object_.find('bndbox').find('xmin').text), int(object_.find('bndbox').find('ymin').text),
                   int(object_.find('bndbox').find('xmax').text), int(object_.find('bndbox').find('ymax').text)]
            kek_objects.append(
                KEKObject(class_id, class_name, KEKBox(box), pv_metadata)
            )
        return KEKFormat(kek_objects, image_metadata)

    def _from_ms_coco(self) -> KEKFormat:
        pass

    def convert(self, *args, **kwargs) -> KEKFormat:
        return {
            'darknet': self._from_darknet,
            'pascalvoc': self._from_pascal_voc,
            'mscoco': self._from_ms_coco
        }.get(self.annotation, self.raise_not_implemented)(*args, **kwargs)


class FromKEKFormatConverter(BaseConverter):
    def __init__(self, annotation: str):
        super().__init__(annotation)

    def _to_darknet(self, kek_format: KEKFormat) -> List[str]:
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

    def _to_pascal_voc(self, kek_format: KEKFormat) -> str:
        root = ET.Element('annotation')
        for k, v in kek_format.image_metadata.items():
            if k == 'database':
                src = ET.SubElement(root, 'source')
                database = ET.SubElement(src, 'database')
                database.text = v
            elif k in ('image_width', 'image_height', 'image_depth'):
                if not root.find('size'):
                    size = ET.SubElement(root, 'size')
                sub = ET.SubElement(size, k.split('_')[-1])
                sub.text = str(v)
            else:
                sub = ET.SubElement(root, k)
                sub.text = str(v)
        for kek_object in kek_format.kek_objects:
            object_ = ET.SubElement(root, 'object')
            sub = ET.SubElement(object_, 'name')
            sub.text = kek_object.class_name
            for k, v in kek_object.pv_metadata.items():
                sub = ET.SubElement(object_, k)
                sub.text = str(v)
            bndbox = ET.SubElement(object_, 'bndbox')
            kek_box = [kek_object.kek_box.top_left_x, kek_object.kek_box.top_left_y,
                       kek_object.kek_box.bottom_right_x, kek_object.kek_box.bottom_right_y]
            for tag, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), kek_box):
                sub = ET.SubElement(bndbox, tag)
                sub.text = str(value)
        xml_string = parseString(ET.tostring(root)).toprettyxml()
        # First line would be '<?xml version="1.0"?>' but it's not needed.
        without_declaration = '\n'.join(xml_string.split('\n')[1:])
        return without_declaration

    def _to_ms_coco(self):
        pass

    def convert(self, *args, **kwargs):
        return {
            'darknet': self._to_darknet,
            'pascalvoc': self._to_pascal_voc,
            'mscoco': self._to_ms_coco
        }.get(self.annotation, self.raise_not_implemented)(*args, **kwargs)
