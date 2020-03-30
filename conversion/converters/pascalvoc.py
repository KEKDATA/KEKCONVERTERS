import os
from typing import Dict
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString


from conversion.entities import KEKBox, KEKObject, KEKFormat


def pascalvoc2kek(image: os.DirEntry, class_mapper: Dict[str, int],
                  base_annotation_path: str = None) -> KEKFormat:
    """
    Converts PASCAL VOC annotation format for given image to KEKFormat representation.

    :param image: Source image;
    :param class_mapper: Mapping for string labels to integer labels.
                         For example, {'car': 0, 'person': 1, ...};
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
        class_id = class_mapper[class_name]
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


def kek2pascalvoc(kek_format: KEKFormat) -> str:
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
