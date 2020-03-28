"""This module contains classes and functions that do intermediate object KEK-representation
of different annotation formats like MS COCO, PASCAL VOC, Darknet, etc."""
from typing import Iterable, List, Union, Tuple


class KEKBox:
    """Class that describes the representation of bounding boxes in KEK-format. KEK-format is:

        [top left x, top left y, bottom right x, bottom right y]

    Yea, it's simple a PASCAL VOC bounding-box format."""
    def __init__(self, box: Iterable[Union[float, int]]) -> None:
        """
        :param box: Bounding-box in some format.
        """
        self.box = tuple(box)

    def __repr__(self) -> str:
        str_box = '[{}]'.format(','.join(map(str, self.box)))
        return str_box

    @classmethod
    def from_darknet(cls, box: Union[Iterable[float], str], image_shape: Tuple[int, int, int]) -> 'KEKBox':
        """
        Makes KEKBox from darknet box representation.

        :param box: Darknet box;
        :param image_shape: Shape of image. Usually (image height, image width, image depth).

        :return: KEKBox.
        """
        def convert(scaled_float_box: Iterable[float], image_w: int,
                    image_h: int) -> Tuple[int, int, int, int]:
            """
            Converts darknet box coordinates to KEK box coordinates.

            :param scaled_float_box: Darknet box coordinates;
            :param image_w: Width of image;
            :param image_h: Height of image.

            :return: KEK box coordinates.
            """
            cx, cy, w, h = scaled_float_box
            tlx = int((cx - w / 2.) * image_w)
            tly = int((cy - h / 2.) * image_h)
            brx = int((cx + w / 2.) * image_w)
            bry = int((cy + h / 2.) * image_h)

            # Sometimes shit happens, guys.
            if tlx < 0:
                tlx = 0
            if tly < 0:
                tly = 0
            if brx > image_width:
                brx = image_width
            if bry > image_height:
                bry = image_height

            return tlx, tly, brx, bry

        image_height, image_width, _ = image_shape
        if isinstance(box, str):
            # Darknet box might comes from string, for example 'class_id xc yc w h' - format.
            return cls(convert(list(map(float, box.split(' ')))[1:], image_width, image_height))
        else:
            return cls(convert(box[1:], image_width, image_height))

    @classmethod
    def from_voc(cls, box: Iterable[int]) -> 'KEKBox':
        """
        Make KEKBox from PASCAL VOC box representation.

        :param box: PASCAL VOC box.

        :return: KEKBox.
        """
        return cls(box)

    @ classmethod
    def from_coco(cls, box: Iterable[float]) -> 'KEKBox':
        """
        Makes KEKBox from MS COCO box representation.

        :param box: MS COCO box.

        :return: KEKBox.
        """
        tlx, tly, w, h = box
        return cls((int(tlx), int(tly), int(tlx + w), int(tly + h)))

    def to_darknet_box(self, image_shape: Tuple[int, int, int]) -> Tuple[float, float, float, float]:
        """
        Converts KEKBox to Darknet box.

        :param image_shape: Shape of image. Usually (image height, image width, image depth).

        :return: Darknet scaled box.
        """
        tlx, tly, brx, bry = self.box
        image_height, image_width, _ = image_shape
        dw = 1. / image_width
        dh = 1. / image_height
        xc = dw * (tlx + brx) / 2.
        yc = dh * (tly + bry) / 2.
        bw = dw * (brx - tlx)
        bh = dh * (bry - tly)
        return xc, yc, bw, bh

    def to_voc_box(self) -> Tuple[int, int, int, int]:
        """
        Converts KEKBox to PASCAL VOC box.

        :return: PASCAL VOC box.
        """
        tlx, tly, brx, bry = self.box
        return tlx, tly, brx, bry

    def to_coco_box(self) -> Tuple[int, int, int, int]:
        """
        Converts KEKBox to MS COCO box.

        :return: MS COCO box.
        """
        tlx, tly, brx, bry = self.box
        return tlx, tly, brx - tlx, bry - tly


class KEKObject:
    """Describes one object on one image.

    PASCAL VOC annotation format has some metadata about object:
    - truncated;
    - difficult;
    - pose;
    - name;
    etc.

    MS COCO - same.

    Darknet has no object metadata except class id.
    """
    def __init__(self, class_id: int, class_name: str, kek_box: Iterable[KEKBox],
                 pascal_voc_metadata: dict = None, ms_coco_metadata: dict = None ) -> None:
        """
        :param class_id: Integer label for class;
        :param class_name: String label for class;
        :param kek_box: List of kek-boxes;
        :param pascal_voc_metadata: Dictionary with metadata specified for PASCAL VOC;
        :param ms_coco_metadata: Dictionary with metadata specified for MS COCO.
        """
        self.class_id = class_id
        self.class_name = class_name
        self.kek_box = kek_box
        self.pv_metadata = pascal_voc_metadata
        self.mc_metadata = ms_coco_metadata


class KEKFormat:
    """Describes all objects on one image and image metadata."""
    def __init__(self, kek_objects: Iterable[KEKObject], image_metadata: dict = None) -> None:
        """
        :param kek_objects: List of KEK-objects;
        :param image_metadata: Dictionary with image's metadata.
        """
        self.kek_objects = kek_objects
        self.image_metadata = image_metadata
