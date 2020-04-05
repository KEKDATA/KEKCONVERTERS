"""This module contains classes and functions that do intermediate object KEK-
representation of different annotation formats like MS COCO, PASCAL VOC, Dark -
net, etc."""
from decimal import Decimal
from typing import Iterable, Union, Tuple


class KEKBox:
    """A class that describes the bounding-box representation in the KEK format:

                [top left x, top left y, bottom right x, bottom right y]
    """
    def __init__(self, top_left_x: int, top_left_y: int, bottom_right_x: int,
                 bottom_right_y: int) -> None:
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y

    @classmethod
    def from_darknet(cls, box: Union[Iterable[float], str],
                     image_shape: Tuple[int, int, int]) -> 'KEKBox':
        """
        Makes KEKBox from darknet box representation.

        :param box: Darknet box;
        :param image_shape: Shape of image. Note that it assumes that image
        has it's format in the following dimensions sequence:

                    (image width, image height, image depth).

        :return: KEKBox.
        """
        def convert(str_box: str, image_width: int,
                    image_height: int) -> Tuple[int, int, int, int]:
            """
            Converts darknet box coordinates to KEK box coordinates.

            :param str_box: Darknet box coordinates;
            :param image_width: Width of image;
            :param image_height: Height of image.

            :return: KEK box coordinates.
            """
            center_x, center_y, box_width, box_height = map(Decimal,
                                                            str_box.split(' '))
            denominator = Decimal('2.')
            scaled_top_left_x = center_x - box_width / denominator
            top_left_x = int(scaled_top_left_x * image_width)
            scaled_top_left_y = center_y - box_height / denominator
            top_left_y = int(scaled_top_left_y * image_height)
            scaled_bottom_right_x = center_x + box_width / denominator
            bottom_right_x = int(scaled_bottom_right_x * image_width)
            scaled_bottom_right_y = center_y + box_height / denominator
            bottom_right_y = int(scaled_bottom_right_y * image_height)

            # Sometimes shit happens, guys.
            if top_left_x < 0:
                top_left_x = 0
            if top_left_y < 0:
                top_left_y = 0
            if bottom_right_x > image_width:
                bottom_right_x = image_width
            if bottom_right_y > image_height:
                bottom_right_y = image_height

            return top_left_x, top_left_y, bottom_right_x, bottom_right_y

        image_width, image_height, _ = image_shape
        if not isinstance(box, str):
            # Sometimes it's float.
            return cls(*convert(' '.join((map(str, box))), image_width,
                                image_height))
        else:
            return cls(*convert(box, image_width, image_height))

    @classmethod
    def from_voc(cls, box: Iterable[int]) -> 'KEKBox':
        return cls(*box)

    @ classmethod
    def from_coco(cls, box: Iterable[float]) -> 'KEKBox':
        top_left_x, top_left_y, box_width, box_height = box
        return cls(top_left_x, top_left_y, top_left_x + box_width,
                   top_left_y + box_height)

    def to_darknet_box(self, image_shape: Tuple[int, int, int]) -> \
            Tuple[float, float, float, float]:
        """
        Converts KEKBox to Darknet box.

        :param image_shape: Shape of image. Note that it assumes that image
        has it's format in the following dimensions sequence:

                    (image width, image height, image depth).

        :return: Darknet scaled box.
        """
        image_width, image_height, _ = image_shape
        nominator = Decimal('1.')
        denominator = Decimal('2.')
        width_scale_factor = nominator / image_width
        height_scale_factor = nominator / image_height

        center_x = (self.top_left_x + self.bottom_right_x) / denominator - 1
        scaled_center_x = width_scale_factor * center_x
        center_y = (self.top_left_y + self.bottom_right_y) / denominator - 1
        scaled_center_y = height_scale_factor * center_y
        box_width = self.bottom_right_x - self.top_left_x
        scaled_box_width = width_scale_factor * box_width
        box_height = self.bottom_right_y - self.top_left_y
        scaled_box_height = height_scale_factor * box_height
        return (float(scaled_center_x), float(scaled_center_y),
                float(scaled_box_width), float(scaled_box_height))

    def to_voc_box(self) -> Tuple[int, int, int, int]:
        return (self.top_left_x, self.top_left_y, self.bottom_right_x,
                self.bottom_right_y)

    def to_coco_box(self) -> Tuple[int, int, int, int]:
        box_width = self.bottom_right_x - self.top_left_x
        box_height = self.bottom_right_y - self.top_left_y
        return self.top_left_x, self.top_left_y, box_width, box_height


class KEKObject:
    """Describes one object on one image. Main information about object is
    stored in class's fields. Additional information (PASCAL VOC <truncated>,
    MS COCO 'license', etc.) is stored in additional_data field in dictionary.
    For more information about KEKObject see README.md."""
    def __init__(self, class_id: int, class_name: str, kek_box: KEKBox,
                 object_additional_data: dict = None) -> None:
        """
        :param class_id: Integer label for class;
        :param class_name: String label for class;
        :param kek_box: Bounding-box in KEKBox format;
        :param object_additional_data: Dictionary with additional data about
        object. See README.md file for description.
        """
        self.class_id = class_id
        self.class_name = class_name
        self.kek_box = kek_box
        self.additional_data = object_additional_data


class KEKImage:
    """Describes one image and all objects on it. Main information about
    image is stored in class's fields. Additional information (PASCAL VOC
    <segmented>, MSCOCO 'license', etc.) is stored in additional_data field.
    For more information about KEKImage see README.md."""
    def __init__(self, id_: int, filename: str, image_shape: Tuple,
                 kek_objects: Iterable[KEKObject],
                 image_additional_data: dict = None) -> None:
        self.id_ = id_
        self.filename = filename
        self.shape = image_shape
        self.kek_objects = kek_objects
        self.additional_data = image_additional_data
